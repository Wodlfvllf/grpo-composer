"""
Patch veRL's collective_rpc for vLLM 0.11.2 compatibility.

vLLM 0.11.2 calls collective_rpc(non_block=True) expecting Future objects,
but veRL 0.7.0's implementation doesn't handle non_block, returning None.
This causes: AttributeError: 'NoneType' object has no attribute 'result'
See: https://github.com/verl-project/verl/issues/4308

This script patches the installed veRL package directly.
Run during Docker/Modal image build AFTER pip installing veRL.
"""

import ast
import importlib.util
import textwrap

PATCHED_METHOD = textwrap.dedent('''\
    def collective_rpc(
        self,
        method: str | Callable,
        timeout: Optional[float] = None,
        args: tuple = (),
        kwargs: Optional[dict[str, Any]] = None,
        non_block: bool = False,
        **kwargs_extra: Any,
    ) -> list[Any]:
        if isinstance(method, str):
            sent_method = method
        else:
            sent_method = pickle.dumps(method)
        del method

        message = pickle.dumps((sent_method, args, kwargs or {}))
        for socket in self.sockets:
            socket.send(message, zmq.DONTWAIT)

        if non_block:
            futures = []
            for socket in self.sockets:
                future = Future()

                def _recv_async(sock=socket, fut=future):
                    try:
                        output = pickle.loads(sock.recv())
                        if isinstance(output, Exception):
                            fut.set_exception(output)
                        else:
                            fut.set_result(output)
                    except Exception as e:
                        fut.set_exception(e)

                thread = threading.Thread(target=_recv_async)
                thread.daemon = True
                thread.start()
                futures.append(future)
            return futures
        else:
            outputs = []
            for socket in self.sockets:
                outputs.append(pickle.loads(socket.recv()))
            for output in outputs:
                if isinstance(output, Exception):
                    raise output
            return outputs
''')

EXTRA_IMPORTS = (
    "import threading\n"
    "from concurrent.futures import Future\n"
)


def patch():
    spec = importlib.util.find_spec(
        "verl.workers.rollout.vllm_rollout.vllm_async_server"
    )
    if spec is None or spec.origin is None:
        print("verl vllm_async_server not found, skipping patch")
        return

    filepath = spec.origin
    with open(filepath, "r") as f:
        source = f.read()

    # Check if already patched
    if "non_block: bool = False" in source:
        print(f"Already patched: {filepath}")
        return

    # Add extra imports after existing imports
    if "import threading" not in source:
        # Insert after "import pickle" line
        source = source.replace("import pickle\n", "import pickle\n" + EXTRA_IMPORTS)

    # Find and replace the collective_rpc method
    # The old method signature starts with "    def collective_rpc(" and ends before "    def check_health"
    old_start = source.find("    def collective_rpc(")
    old_end = source.find("    def check_health(")

    if old_start == -1:
        print(f"Could not find collective_rpc method in {filepath}")
        return

    if old_end == -1:
        # Fallback: find next method or end of class
        old_end = source.find("\n    def ", old_start + 10)
        if old_end == -1:
            print(f"Could not find end of collective_rpc method in {filepath}")
            return

    # Indent the patched method to match class indentation (4 spaces)
    indented_patch = textwrap.indent(PATCHED_METHOD, "    ")

    source = source[:old_start] + indented_patch + "\n" + source[old_end:]

    with open(filepath, "w") as f:
        f.write(source)

    print(f"Successfully patched: {filepath}")


if __name__ == "__main__":
    patch()
