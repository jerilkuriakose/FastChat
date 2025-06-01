"""
A model worker that executes the model based on vLLM.

See documentations at docs/vllm_integration.md
"""

# =============================================================================
# IMPROVED FIX: Comprehensive monkey patch for vLLM argparse compatibility
# This must be the very first thing before any other imports
# =============================================================================
import argparse
import inspect

# Store original methods
_original_add_argument = argparse.ArgumentParser.add_argument

# Get all Action classes that might need patching
action_classes = []
for name in dir(argparse):
    obj = getattr(argparse, name)
    if (
        inspect.isclass(obj)
        and issubclass(obj, argparse.Action)
        and obj != argparse.Action
    ):
        action_classes.append(obj)

# Store original __init__ methods for all Action classes
original_inits = {}
for action_class in action_classes:
    if hasattr(action_class, "__init__"):
        original_inits[action_class] = action_class.__init__


def _patched_add_argument(self, *args, **kwargs):
    """Remove unsupported argparse arguments that newer vLLM versions use"""
    # Extended list of potentially unsupported arguments
    unsupported_args = [
        "deprecated",
        "ge",
        "le",
        "lt",
        "gt",
        "min_length",
        "max_length",
        "exit_on_error",
    ]
    for arg in unsupported_args:
        if arg in kwargs:
            print(f"Warning: Removing unsupported argparse argument '{arg}'")
            del kwargs[arg]
    return _original_add_argument(self, *args, **kwargs)


def create_patched_init(original_init, class_name):
    """Create a patched __init__ method for an Action class"""

    def _patched_init(self, *args, **kwargs):
        unsupported_args = [
            "deprecated",
            "ge",
            "le",
            "lt",
            "gt",
            "min_length",
            "max_length",
            "exit_on_error",
        ]
        for arg in unsupported_args:
            if arg in kwargs:
                print(f"Warning: Removing unsupported {class_name} argument '{arg}'")
                del kwargs[arg]
        return original_init(self, *args, **kwargs)

    return _patched_init


# Apply patches
argparse.ArgumentParser.add_argument = _patched_add_argument

# Patch all Action subclass __init__ methods
for action_class in action_classes:
    if action_class in original_inits:
        action_class.__init__ = create_patched_init(
            original_inits[action_class], action_class.__name__
        )

print("Applied comprehensive argparse compatibility patches for vLLM")
# =============================================================================
import asyncio
import json
from typing import List

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn
from vllm import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid

from fastchat.serve.base_model_worker import BaseModelWorker
from fastchat.serve.model_worker import (
    logger,
    worker_id,
)
from fastchat.utils import get_context_length, is_partial_stop

app = FastAPI()


class VLLMWorker(BaseModelWorker):
    def __init__(
        self,
        controller_addr: str,
        worker_addr: str,
        worker_id: str,
        model_path: str,
        model_names: List[str],
        limit_worker_concurrency: int,
        no_register: bool,
        llm_engine: AsyncLLMEngine,
        conv_template: str,
    ):
        super().__init__(
            controller_addr,
            worker_addr,
            worker_id,
            model_path,
            model_names,
            limit_worker_concurrency,
            conv_template,
        )

        logger.info(
            f"Loading the model {self.model_names} on worker {worker_id}, worker type: vLLM worker..."
        )

        # Store the engine for later async access
        self.llm_engine = llm_engine
        self.tokenizer = None
        self.context_len = None
        self._initialization_complete = False
        self._initialization_started = False

        if not no_register:
            self.init_heart_beat()

    async def _initialize_async(self):
        """Initialize tokenizer and context length asynchronously"""
        try:
            # Try the new async API methods first
            if hasattr(self.llm_engine, "get_tokenizer"):
                self.tokenizer = await self.llm_engine.get_tokenizer()
                logger.info("Successfully obtained tokenizer using async API")

            if hasattr(self.llm_engine, "get_model_config"):
                model_config = await self.llm_engine.get_model_config()
                self.context_len = get_context_length(model_config.hf_config)
                logger.info(
                    f"Successfully obtained model config, context length: {self.context_len}"
                )

            # Fallback methods for different vLLM versions
            if self.tokenizer is None or self.context_len is None:
                await self._fallback_initialization()

            logger.info("VLLMWorker initialization completed successfully")

        except Exception as e:
            logger.error(f"Failed to initialize VLLMWorker: {e}")
            await self._fallback_initialization()
        finally:
            self._initialization_complete = True

    async def _fallback_initialization(self):
        """Fallback initialization methods for older vLLM versions"""
        try:
            # Try accessing engine directly (older vLLM versions)
            if hasattr(self.llm_engine, "engine"):
                if self.tokenizer is None:
                    tokenizer = self.llm_engine.engine.tokenizer
                    # Handle TokenizerGroup wrapper (vLLM >= 0.2.7)
                    if hasattr(tokenizer, "tokenizer"):
                        self.tokenizer = tokenizer.tokenizer
                    else:
                        self.tokenizer = tokenizer
                    logger.info("Fallback: obtained tokenizer from engine.tokenizer")

                if self.context_len is None:
                    model_config = self.llm_engine.engine.model_config
                    self.context_len = get_context_length(model_config.hf_config)
                    logger.info(
                        f"Fallback: obtained context length from engine: {self.context_len}"
                    )

            # Try other possible access patterns
            elif hasattr(self.llm_engine, "_engine"):
                if self.tokenizer is None:
                    tokenizer = self.llm_engine._engine.tokenizer
                    if hasattr(tokenizer, "tokenizer"):
                        self.tokenizer = tokenizer.tokenizer
                    else:
                        self.tokenizer = tokenizer
                    logger.info("Fallback: obtained tokenizer from _engine.tokenizer")

                if self.context_len is None:
                    model_config = self.llm_engine._engine.model_config
                    self.context_len = get_context_length(model_config.hf_config)
                    logger.info(
                        f"Fallback: obtained context length from _engine: {self.context_len}"
                    )

            else:
                logger.warning(
                    "Unable to access tokenizer or model config through any known method"
                )
                # Set reasonable defaults
                if self.tokenizer is None:
                    logger.warning(
                        "Tokenizer not available - some features may not work"
                    )
                if self.context_len is None:
                    self.context_len = 2048  # Conservative default
                    logger.warning(f"Using default context length: {self.context_len}")

        except Exception as fallback_error:
            logger.error(f"Fallback initialization also failed: {fallback_error}")
            # Set minimal defaults to prevent crashes
            if self.context_len is None:
                self.context_len = 2048
                logger.warning("Using minimal default context length: 2048")

    async def get_tokenizer(self):
        """Get tokenizer, initializing if necessary"""
        # Lazy initialization - only initialize when first needed
        if not self._initialization_started:
            self._initialization_started = True
            await self._initialize_async()

        # Wait for initialization to complete if it's still in progress
        max_wait = 30  # seconds
        wait_time = 0
        while not self._initialization_complete and wait_time < max_wait:
            await asyncio.sleep(0.1)
            wait_time += 0.1

        if not self._initialization_complete:
            logger.warning("Tokenizer initialization did not complete in time")

        return self.tokenizer

    async def generate_stream(self, params):
        self.call_ct += 1

        context = params.pop("prompt")
        request_id = params.pop("request_id")
        temperature = float(params.get("temperature", 1.0))
        top_p = float(params.get("top_p", 1.0))
        top_k = params.get("top_k", -1)
        presence_penalty = float(params.get("presence_penalty", 0.0))
        frequency_penalty = float(params.get("frequency_penalty", 0.0))
        max_new_tokens = params.get("max_new_tokens", 256)
        stop_str = params.get("stop", None)
        stop_token_ids = params.get("stop_token_ids", None) or []

        # Get tokenizer asynchronously
        tokenizer = await self.get_tokenizer()
        if (
            tokenizer
            and hasattr(tokenizer, "eos_token_id")
            and tokenizer.eos_token_id is not None
        ):
            stop_token_ids.append(tokenizer.eos_token_id)

        echo = params.get("echo", True)
        use_beam_search = params.get("use_beam_search", False)
        best_of = params.get("best_of", None)

        request = params.get("request", None)

        # Handle stop_str
        stop = set()
        if isinstance(stop_str, str) and stop_str != "":
            stop.add(stop_str)
        elif isinstance(stop_str, list) and stop_str != []:
            stop.update(stop_str)

        if tokenizer:
            for tid in stop_token_ids:
                if tid is not None:
                    try:
                        s = tokenizer.decode(tid)
                        if s != "":
                            stop.add(s)
                    except Exception as e:
                        logger.warning(f"Failed to decode stop token {tid}: {e}")

        # make sampling params in vllm
        top_p = max(top_p, 1e-5)
        if temperature <= 1e-5:
            top_p = 1.0

        # Create sampling params with error handling for version compatibility
        sampling_params_kwargs = {
            "n": 1,
            "temperature": temperature,
            "top_p": top_p,
            "stop": list(stop),
            "stop_token_ids": stop_token_ids,
            "max_tokens": max_new_tokens,
            "top_k": top_k,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "seed": 1234,
            "skip_special_tokens": False,
        }

        # Add optional parameters that might not exist in all vLLM versions
        if best_of is not None:
            sampling_params_kwargs["best_of"] = best_of

        # Remove use_beam_search if not supported (newer vLLM versions)
        try:
            sampling_params = SamplingParams(**sampling_params_kwargs)
        except TypeError as e:
            if "use_beam_search" in str(e):
                # Remove use_beam_search and try again
                logger.warning(
                    "use_beam_search parameter not supported in this vLLM version, ignoring it"
                )
                sampling_params = SamplingParams(**sampling_params_kwargs)
            else:
                raise e

        results_generator = engine.generate(context, sampling_params, request_id)

        async for request_output in results_generator:
            prompt = request_output.prompt
            if echo:
                text_outputs = [
                    prompt + output.text for output in request_output.outputs
                ]
            else:
                text_outputs = [output.text for output in request_output.outputs]
            text_outputs = " ".join(text_outputs)

            partial_stop = any(is_partial_stop(text_outputs, i) for i in stop)
            # prevent yielding partial stop sequence
            if partial_stop:
                continue

            aborted = False
            if request and await request.is_disconnected():
                await engine.abort(request_id)
                request_output.finished = True
                aborted = True
                for output in request_output.outputs:
                    output.finish_reason = "abort"

            prompt_tokens = len(request_output.prompt_token_ids)
            completion_tokens = sum(
                len(output.token_ids) for output in request_output.outputs
            )
            ret = {
                "text": text_outputs,
                "error_code": 0,
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
                "cumulative_logprob": [
                    output.cumulative_logprob for output in request_output.outputs
                ],
                "finish_reason": request_output.outputs[0].finish_reason
                if len(request_output.outputs) == 1
                else [output.finish_reason for output in request_output.outputs],
            }
            # Emit twice here to ensure a 'finish_reason' with empty content in the OpenAI API response.
            # This aligns with the behavior of model_worker.
            if request_output.finished:
                yield (json.dumps({**ret, **{"finish_reason": None}}) + "\0").encode()
            yield (json.dumps(ret) + "\0").encode()

            if aborted:
                break

    async def generate(self, params):
        async for x in self.generate_stream(params):
            pass
        return json.loads(x[:-1].decode())


def release_worker_semaphore():
    worker.semaphore.release()


def acquire_worker_semaphore():
    if worker.semaphore is None:
        worker.semaphore = asyncio.Semaphore(worker.limit_worker_concurrency)
    return worker.semaphore.acquire()


def create_background_tasks(request_id):
    async def abort_request() -> None:
        await engine.abort(request_id)

    background_tasks = BackgroundTasks()
    background_tasks.add_task(release_worker_semaphore)
    background_tasks.add_task(abort_request)
    return background_tasks


@app.post("/worker_generate_stream")
async def api_generate_stream(request: Request):
    params = await request.json()
    await acquire_worker_semaphore()
    request_id = random_uuid()
    params["request_id"] = request_id
    params["request"] = request
    generator = worker.generate_stream(params)
    background_tasks = create_background_tasks(request_id)
    return StreamingResponse(generator, background=background_tasks)


@app.post("/worker_generate")
async def api_generate(request: Request):
    params = await request.json()
    await acquire_worker_semaphore()
    request_id = random_uuid()
    params["request_id"] = request_id
    params["request"] = request
    output = await worker.generate(params)
    release_worker_semaphore()
    await engine.abort(request_id)
    return JSONResponse(output)


@app.post("/worker_get_status")
async def api_get_status(request: Request):
    return worker.get_status()


@app.post("/count_token")
async def api_count_token(request: Request):
    params = await request.json()
    return worker.count_token(params)


@app.post("/worker_get_conv_template")
async def api_get_conv(request: Request):
    return worker.get_conv_template()


@app.post("/model_details")
async def api_model_details(request: Request):
    # Ensure context_len is available using lazy initialization
    if not worker._initialization_started:
        worker._initialization_started = True
        await worker._initialize_async()

    # Wait for initialization if needed
    max_wait = 30  # seconds
    wait_time = 0
    while not worker._initialization_complete and wait_time < max_wait:
        await asyncio.sleep(0.1)
        wait_time += 0.1

    return {"context_length": worker.context_len or 2048}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=21002)
    parser.add_argument("--worker-address", type=str, default="http://localhost:21002")
    parser.add_argument(
        "--controller-address", type=str, default="http://localhost:21001"
    )
    parser.add_argument("--model-path", type=str, default="lmsys/vicuna-7b-v1.5")
    parser.add_argument(
        "--model-names",
        type=lambda s: s.split(","),
        help="Optional display comma separated names",
    )
    parser.add_argument("--limit-worker-concurrency", type=int, default=1024)
    parser.add_argument("--no-register", action="store_true")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument(
        "--conv-template", type=str, default=None, help="Conversation prompt template."
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_false",
        default=True,
        help="Trust remote code (e.g., from HuggingFace) when"
        "downloading the model and tokenizer.",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
        help="The ratio (between 0 and 1) of GPU memory to"
        "reserve for the model weights, activations, and KV cache. Higher"
        "values will increase the KV cache size and thus improve the model's"
        "throughput. However, if the value is too high, it may cause out-of-"
        "memory (OOM) errors.",
    )

    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    if args.model_path:
        args.model = args.model_path
    if args.num_gpus > 1:
        args.tensor_parallel_size = args.num_gpus

    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    worker = VLLMWorker(
        args.controller_address,
        args.worker_address,
        worker_id,
        args.model_path,
        args.model_names,
        args.limit_worker_concurrency,
        args.no_register,
        engine,
        args.conv_template,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
