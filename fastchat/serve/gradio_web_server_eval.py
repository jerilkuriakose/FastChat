"""
The gradio demo server for evaluation with a single model.
This is a specialized version for human evaluation tasks.
"""

import argparse
from collections import defaultdict
import datetime
import hashlib
import json
import os
import random
import re
import time
import uuid

import gradio as gr
import requests

from fastchat.constants import (
    LOGDIR,
    WORKER_API_TIMEOUT,
    ErrorCode,
    MODERATION_MSG,
    CONVERSATION_LIMIT_MSG,
    RATE_LIMIT_MSG,
    SERVER_ERROR_MSG,
    INPUT_CHAR_LEN_LIMIT,
    CONVERSATION_TURN_LIMIT,
    SESSION_EXPIRATION_TIME,
)
from fastchat.model.model_adapter import (
    get_conversation_template,
)
from fastchat.model.model_registry import get_model_info, model_info
from fastchat.serve.api_provider import get_api_provider_stream_iter
from fastchat.serve.remote_logger import get_remote_logger
from fastchat.utils import (
    build_logger,
    get_window_url_params_js,
    get_window_url_params_with_tos_js,
    moderation_filter,
    parse_gradio_auth_creds,
    load_image,
)

# Import shared functions from the main gradio_web_server
from fastchat.serve.gradio_web_server import (
    State,
    get_conv_log_filename,
    get_ip,
    report_csam_image,
    _prepare_text_with_image,
    model_worker_stream_iter,
    is_limit_reached,
    block_css,
    get_model_description_md,
    build_about,
    acknowledgment_md,
    no_change_btn,
    enable_btn,
    disable_btn,
    invisible_btn,
)

logger = build_logger("gradio_web_server_eval", "gradio_web_server_eval.log")

headers = {"User-Agent": "FastChat Client Eval"}

# Separate global variables for evaluation server
controller_url_eval = None
enable_moderation_eval = False
use_remote_storage_eval = False
api_endpoint_info_eval = {}
invisible_flag_btn = gr.update(interactive=False, visible=False)


def set_global_vars_eval(controller_url_, enable_moderation_, use_remote_storage_):
    """Set global variables for evaluation server"""
    global controller_url_eval, enable_moderation_eval, use_remote_storage_eval
    controller_url_eval = controller_url_
    enable_moderation_eval = enable_moderation_
    use_remote_storage_eval = use_remote_storage_


def load_api_endpoint_info_eval(register_api_endpoint_file):
    """Load API endpoint information for evaluation server"""
    global api_endpoint_info_eval
    if register_api_endpoint_file:
        api_endpoint_info_eval = json.load(open(register_api_endpoint_file))
        logger.info(
            f"Loaded API endpoints for evaluation: {list(api_endpoint_info_eval.keys())}"
        )
    else:
        api_endpoint_info_eval = {}


def vote_last_response_eval(
    state,
    vote_type,
    model_selector,
    request: gr.Request,
    feedback="",
    username="",
    question_id="",
    evaluation_checkboxes=None,  # New parameter for checkboxes
):
    """Enhanced voting for evaluation with question tracking and checkboxes"""
    filename = get_conv_log_filename()
    if "llava" in model_selector:
        filename = filename.replace("2024", "vision-eval-tmp-2024")
    else:
        # Add eval prefix to log files
        filename = filename.replace("conv.json", "eval-conv.json")

    # Process checkbox data
    selected_issues = evaluation_checkboxes if evaluation_checkboxes else []

    with open(filename, "a") as fout:
        data = {
            "tstamp": round(time.time(), 4),
            "type": vote_type,
            "model": model_selector,
            "state": state.dict(),
            "ip": get_ip(request),
            "feedback": feedback,
            "username": username,
            "question_id": question_id,
            "evaluation_mode": True,
            "evaluation_issues": selected_issues,  # Add checkbox data to logs
        }
        fout.write(json.dumps(data) + "\n")
    get_remote_logger().log(data)


def upvote_last_response_eval(
    state,
    model_selector,
    tb_feedback,
    tb_username,
    tb_question_id,
    request: gr.Request,  # Moved before *evaluation_checkboxes
    *evaluation_checkboxes,
):
    ip = get_ip(request)
    logger.info(
        f"eval upvote. ip: {ip} | username: {tb_username} | question_id: {tb_question_id} | issues: {evaluation_checkboxes}"
    )
    vote_last_response_eval(
        state,
        "upvote",
        model_selector,
        request,
        feedback=tb_feedback.strip(),
        username=tb_username,
        question_id=tb_question_id.strip(),
        evaluation_checkboxes=list(evaluation_checkboxes),
    )
    return (
        "",  # textbox
        disable_btn,  # upvote button
        disable_btn,  # downvote button
        disable_btn,  # flag button
        gr.update(value="", visible=False, interactive=True),  # textbox_feedback
        invisible_btn,  # textbox_feedback button
        gr.update(visible=False),  # checkbox_label
        gr.update(value=False, visible=False),  # checkbox_factually_incorrect
        gr.update(value=False, visible=False),  # checkbox_no_answer
        gr.update(value=False, visible=False),  # checkbox_outdated_info
        gr.update(value=False, visible=False),  # checkbox_hallucination
        gr.update(value=False, visible=False),  # checkbox_prompt_misunderstood
        gr.update(value=False, visible=False),  # checkbox_off_topic
        gr.update(value=False, visible=False),  # checkbox_greeting_ignored
        gr.update(value=False, visible=False),  # checkbox_repetitive
        gr.update(value=False, visible=False),  # checkbox_contradiction
        gr.update(value=False, visible=False),  # checkbox_code_switching
        gr.update(value=False, visible=False),  # checkbox_language_misalignment
        gr.update(value=False, visible=False),  # checkbox_verbosity
        gr.update(value=False, visible=False),  # checkbox_brief_answer
    )


def downvote_last_response_eval(
    state,
    model_selector,
    tb_feedback,
    tb_username,
    tb_question_id,
    request: gr.Request,  # Moved before *evaluation_checkboxes
    *evaluation_checkboxes,
):
    ip = get_ip(request)
    logger.info(
        f"eval downvote. ip: {ip} | username: {tb_username} | question_id: {tb_question_id} | issues: {evaluation_checkboxes}"
    )
    vote_last_response_eval(
        state,
        "downvote",
        model_selector,
        request,
        feedback=tb_feedback.strip(),
        username=tb_username,
        question_id=tb_question_id.strip(),
        evaluation_checkboxes=list(evaluation_checkboxes),
    )
    return (
        "",  # textbox
        disable_btn,  # upvote button
        disable_btn,  # downvote button
        disable_btn,  # flag button
        gr.update(value="", visible=False, interactive=True),  # textbox_feedback
        invisible_btn,  # textbox_feedback button
        gr.update(visible=False),  # checkbox_label
        gr.update(value=False, visible=False),  # checkbox_factually_incorrect
        gr.update(value=False, visible=False),  # checkbox_no_answer
        gr.update(value=False, visible=False),  # checkbox_outdated_info
        gr.update(value=False, visible=False),  # checkbox_hallucination
        gr.update(value=False, visible=False),  # checkbox_prompt_misunderstood
        gr.update(value=False, visible=False),  # checkbox_off_topic
        gr.update(value=False, visible=False),  # checkbox_greeting_ignored
        gr.update(value=False, visible=False),  # checkbox_repetitive
        gr.update(value=False, visible=False),  # checkbox_contradiction
        gr.update(value=False, visible=False),  # checkbox_code_switching
        gr.update(value=False, visible=False),  # checkbox_language_misalignment
        gr.update(value=False, visible=False),  # checkbox_verbosity
        gr.update(value=False, visible=False),  # checkbox_brief_answer
    )


def flag_last_response_eval(
    state,
    model_selector,
    tb_feedback,
    tb_username,
    tb_question_id,
    request: gr.Request,  # Moved before *evaluation_checkboxes
    *evaluation_checkboxes,
):
    ip = get_ip(request)
    logger.info(
        f"eval flag. ip: {ip} | username: {tb_username} | question_id: {tb_question_id} | issues: {evaluation_checkboxes}"
    )
    vote_last_response_eval(
        state,
        "flag",
        model_selector,
        request,
        feedback=tb_feedback.strip(),
        username=tb_username,
        question_id=tb_question_id.strip(),
        evaluation_checkboxes=list(evaluation_checkboxes),
    )
    return (
        "",  # textbox
        disable_btn,  # upvote button
        disable_btn,  # downvote button
        disable_btn,  # flag button
        gr.update(value="", visible=False, interactive=True),  # textbox_feedback
        invisible_btn,  # textbox_feedback button
        gr.update(visible=False),  # checkbox_label
        gr.update(value=False, visible=False),  # checkbox_factually_incorrect
        gr.update(value=False, visible=False),  # checkbox_no_answer
        gr.update(value=False, visible=False),  # checkbox_outdated_info
        gr.update(value=False, visible=False),  # checkbox_hallucination
        gr.update(value=False, visible=False),  # checkbox_prompt_misunderstood
        gr.update(value=False, visible=False),  # checkbox_off_topic
        gr.update(value=False, visible=False),  # checkbox_greeting_ignored
        gr.update(value=False, visible=False),  # checkbox_repetitive
        gr.update(value=False, visible=False),  # checkbox_contradiction
        gr.update(value=False, visible=False),  # checkbox_code_switching
        gr.update(value=False, visible=False),  # checkbox_language_misalignment
        gr.update(value=False, visible=False),  # checkbox_verbosity
        gr.update(value=False, visible=False),  # checkbox_brief_answer
    )


# New functions for auto-clearing after vote (single-turn mode)
def upvote_last_response_eval_autoclear(
    state,
    model_selector,
    tb_feedback,
    tb_username,
    tb_question_id,
    request: gr.Request,
    *evaluation_checkboxes,
):
    ip = get_ip(request)
    logger.info(
        f"eval upvote (autoclear). ip: {ip} | username: {tb_username} | question_id: {tb_question_id} | issues: {evaluation_checkboxes}"
    )
    vote_last_response_eval(
        state,
        "upvote",
        model_selector,
        request,
        feedback=tb_feedback.strip(),
        username=tb_username,
        question_id=tb_question_id.strip(),
        evaluation_checkboxes=list(evaluation_checkboxes),
    )
    # Return values to reset UI including state and chatbot (auto-clear)
    return (
        None,  # state - clear conversation
        [],  # chatbot - clear chat history
        "",  # textbox
        None,  # imagebox
        disable_btn,  # upvote_btn
        disable_btn,  # downvote_btn
        disable_btn,  # flag_btn
        disable_btn,  # regenerate_btn
        disable_btn,  # clear_btn
        gr.update(value="", visible=False, interactive=True),  # textbox_feedback
        invisible_btn,  # textbox_feedback button
        gr.update(visible=False),  # checkbox_label
        gr.update(value=False, visible=False),  # checkbox_factually_incorrect
        gr.update(value=False, visible=False),  # checkbox_no_answer
        gr.update(value=False, visible=False),  # checkbox_outdated_info
        gr.update(value=False, visible=False),  # checkbox_hallucination
        gr.update(value=False, visible=False),  # checkbox_prompt_misunderstood
        gr.update(value=False, visible=False),  # checkbox_off_topic
        gr.update(value=False, visible=False),  # checkbox_greeting_ignored
        gr.update(value=False, visible=False),  # checkbox_repetitive
        gr.update(value=False, visible=False),  # checkbox_contradiction
        gr.update(value=False, visible=False),  # checkbox_code_switching
        gr.update(value=False, visible=False),  # checkbox_language_misalignment
        gr.update(value=False, visible=False),  # checkbox_verbosity
        gr.update(value=False, visible=False),  # checkbox_brief_answer
    )


def downvote_last_response_eval_autoclear(
    state,
    model_selector,
    tb_feedback,
    tb_username,
    tb_question_id,
    request: gr.Request,
    *evaluation_checkboxes,
):
    ip = get_ip(request)
    logger.info(
        f"eval downvote (autoclear). ip: {ip} | username: {tb_username} | question_id: {tb_question_id} | issues: {evaluation_checkboxes}"
    )
    vote_last_response_eval(
        state,
        "downvote",
        model_selector,
        request,
        feedback=tb_feedback.strip(),
        username=tb_username,
        question_id=tb_question_id.strip(),
        evaluation_checkboxes=list(evaluation_checkboxes),
    )
    # Return values to reset UI including state and chatbot (auto-clear)
    return (
        None,  # state - clear conversation
        [],  # chatbot - clear chat history
        "",  # textbox
        None,  # imagebox
        disable_btn,  # upvote_btn
        disable_btn,  # downvote_btn
        disable_btn,  # flag_btn
        disable_btn,  # regenerate_btn
        disable_btn,  # clear_btn
        gr.update(value="", visible=False, interactive=True),  # textbox_feedback
        invisible_btn,  # textbox_feedback button
        gr.update(visible=False),  # checkbox_label
        gr.update(value=False, visible=False),  # checkbox_factually_incorrect
        gr.update(value=False, visible=False),  # checkbox_no_answer
        gr.update(value=False, visible=False),  # checkbox_outdated_info
        gr.update(value=False, visible=False),  # checkbox_hallucination
        gr.update(value=False, visible=False),  # checkbox_prompt_misunderstood
        gr.update(value=False, visible=False),  # checkbox_off_topic
        gr.update(value=False, visible=False),  # checkbox_greeting_ignored
        gr.update(value=False, visible=False),  # checkbox_repetitive
        gr.update(value=False, visible=False),  # checkbox_contradiction
        gr.update(value=False, visible=False),  # checkbox_code_switching
        gr.update(value=False, visible=False),  # checkbox_language_misalignment
        gr.update(value=False, visible=False),  # checkbox_verbosity
        gr.update(value=False, visible=False),  # checkbox_brief_answer
    )


def flag_last_response_eval_autoclear(
    state,
    model_selector,
    tb_feedback,
    tb_username,
    tb_question_id,
    request: gr.Request,
    *evaluation_checkboxes,
):
    ip = get_ip(request)
    logger.info(
        f"eval flag (autoclear). ip: {ip} | username: {tb_username} | question_id: {tb_question_id} | issues: {evaluation_checkboxes}"
    )
    vote_last_response_eval(
        state,
        "flag",
        model_selector,
        request,
        feedback=tb_feedback.strip(),
        username=tb_username,
        question_id=tb_question_id.strip(),
        evaluation_checkboxes=list(evaluation_checkboxes),
    )
    # Return values to reset UI including state and chatbot (auto-clear)
    return (
        None,  # state - clear conversation
        [],  # chatbot - clear chat history
        "",  # textbox
        None,  # imagebox
        disable_btn,  # upvote_btn
        disable_btn,  # downvote_btn
        disable_btn,  # flag_btn
        disable_btn,  # regenerate_btn
        disable_btn,  # clear_btn
        gr.update(value="", visible=False, interactive=True),  # textbox_feedback
        invisible_btn,  # textbox_feedback button
        gr.update(visible=False),  # checkbox_label
        gr.update(value=False, visible=False),  # checkbox_factually_incorrect
        gr.update(value=False, visible=False),  # checkbox_no_answer
        gr.update(value=False, visible=False),  # checkbox_outdated_info
        gr.update(value=False, visible=False),  # checkbox_hallucination
        gr.update(value=False, visible=False),  # checkbox_prompt_misunderstood
        gr.update(value=False, visible=False),  # checkbox_off_topic
        gr.update(value=False, visible=False),  # checkbox_greeting_ignored
        gr.update(value=False, visible=False),  # checkbox_repetitive
        gr.update(value=False, visible=False),  # checkbox_contradiction
        gr.update(value=False, visible=False),  # checkbox_code_switching
        gr.update(value=False, visible=False),  # checkbox_language_misalignment
        gr.update(value=False, visible=False),  # checkbox_verbosity
        gr.update(value=False, visible=False),  # checkbox_brief_answer
    )


def regenerate_eval(state, request: gr.Request):
    ip = get_ip(request)
    logger.info(f"eval regenerate. ip: {ip}")
    if not state.regen_support:
        state.skip_next = True
        # Keep flag button invisible
        return (state, state.to_gradio_chatbot(), "", None) + (
            no_change_btn,
            no_change_btn,
            invisible_flag_btn,
            no_change_btn,
            no_change_btn,
        )
    state.conv.update_last_message(None)
    # Keep flag button invisible
    return (state, state.to_gradio_chatbot(), "", None) + (
        disable_btn,
        disable_btn,
        invisible_flag_btn,
        disable_btn,
        disable_btn,
    )


def clear_history_eval(request: gr.Request):
    ip = get_ip(request)
    logger.info(f"eval clear_history. ip: {ip}")
    state = None
    # Reset checkboxes when clearing history and keep flag button invisible
    return (
        (state, [], "", None)
        + (
            disable_btn,
            disable_btn,
            invisible_flag_btn,
            disable_btn,
            disable_btn,
        )  # Keep flag invisible
        + (gr.update(value="", visible=False, interactive=True),)  # textbox_feedback
        + (invisible_btn,)
        + (gr.update(visible=False),)  # checkbox_label
        + (gr.update(value=False, visible=False),) * 13  # all 13 checkboxes
    )


def flash_buttons_eval():
    """Flash buttons for evaluation UI and show checkboxes"""
    btn_updates = [
        [disable_btn] * 2
        + [invisible_flag_btn]
        + [enable_btn] * 2,  # Keep flag invisible
        [enable_btn] * 2
        + [invisible_flag_btn]
        + [enable_btn] * 2,  # Keep flag invisible
    ]

    for i in range(4):
        if i == 3:  # On the last iteration, show checkboxes and feedback
            yield (
                btn_updates[i % 2]
                + [
                    gr.update(visible=True, interactive=True)
                ]  # textbox_feedback - make both visible AND interactive
                + [gr.update(visible=True)]  # checkbox_label
                + [gr.update(value=False, visible=True)] * 13  # all 13 checkboxes
            )
        else:
            yield (
                btn_updates[i % 2]
                + [gr.update(visible=False, interactive=False)]  # textbox_feedback
                + [gr.update(visible=False)]  # checkbox_label
                + [gr.update(value=False, visible=False)] * 13  # all 13 checkboxes
            )
        time.sleep(0.1)


def add_text_eval(state, model_selector, text, image, sys_msg, request: gr.Request):
    ip = get_ip(request)
    logger.info(f"eval add_text. ip: {ip}. len: {len(text)}")

    if state is None:
        state = State(model_selector)

    if len(text) <= 0:
        state.skip_next = True
        return (state, state.to_gradio_chatbot(), "", None) + (no_change_btn,) * 5

    if sys_msg:
        system_message = f"{sys_msg}"
        state.conv.set_system_message(system_message)

    all_conv_text = state.conv.get_prompt()
    all_conv_text = all_conv_text[-2000:] + "\nuser: " + text
    flagged = moderation_filter(all_conv_text, [state.model_name])

    if flagged:
        logger.info(f"eval violate moderation. ip: {ip}. text: {text}")
        text = MODERATION_MSG

    if (len(state.conv.messages) - state.conv.offset) // 2 >= CONVERSATION_TURN_LIMIT:
        logger.info(f"eval conversation turn limit. ip: {ip}. text: {text}")
        state.skip_next = True
        return (state, state.to_gradio_chatbot(), CONVERSATION_LIMIT_MSG, None) + (
            no_change_btn,
        ) * 5

    text = text[:INPUT_CHAR_LEN_LIMIT]  # Hard cut-off
    text = _prepare_text_with_image(state, text, image, csam_flag=False)
    state.conv.append_message(state.conv.roles[0], text)
    state.conv.append_message(state.conv.roles[1], None)
    return (state, state.to_gradio_chatbot(), "", None) + (disable_btn,) * 5


def bot_response_eval(
    state,
    temperature,
    top_p,
    max_new_tokens,
    request: gr.Request,
    apply_rate_limit=True,
    use_recommended_config=False,
):
    """Bot response for evaluation mode"""
    ip = get_ip(request)
    logger.info(f"eval bot_response. ip: {ip}")
    start_tstamp = time.time()
    temperature = float(temperature)
    top_p = float(top_p)
    max_new_tokens = int(max_new_tokens)

    if state.skip_next:
        state.skip_next = False
        # Keep flag button invisible
        yield (state, state.to_gradio_chatbot()) + (
            no_change_btn,
            no_change_btn,
            invisible_flag_btn,
            no_change_btn,
            no_change_btn,
        )
        return

    if apply_rate_limit:
        ret = is_limit_reached(state.model_name, ip)
        if ret is not None and ret["is_limit_reached"]:
            error_msg = RATE_LIMIT_MSG + "\n\n" + ret["reason"]
            logger.info(
                f"eval rate limit reached. ip: {ip}. error_msg: {ret['reason']}"
            )
            state.conv.update_last_message(error_msg)
            # Keep flag button invisible
            yield (state, state.to_gradio_chatbot()) + (
                no_change_btn,
                no_change_btn,
                invisible_flag_btn,
                no_change_btn,
                no_change_btn,
            )
            return

    conv, model_name = state.conv, state.model_name
    model_api_dict = (
        api_endpoint_info_eval[model_name]
        if model_name in api_endpoint_info_eval
        else None
    )
    images = conv.get_images()

    if model_api_dict is None:
        # Query worker address
        ret = requests.post(
            controller_url_eval + "/get_worker_address", json={"model": model_name}
        )
        worker_addr = ret.json()["address"]
        logger.info(f"eval model_name: {model_name}, worker_addr: {worker_addr}")

        # No available worker
        if worker_addr == "":
            conv.update_last_message(SERVER_ERROR_MSG)
            yield (
                state,
                state.to_gradio_chatbot(),
                disable_btn,
                disable_btn,
                invisible_flag_btn,  # Keep flag button invisible
                enable_btn,
                enable_btn,
            )
            return

        # Construct prompt
        prompt = conv.get_prompt()
        # Set repetition_penalty
        if "t5" in model_name:
            repetition_penalty = 1.2
        else:
            repetition_penalty = 1.0

        stream_iter = model_worker_stream_iter(
            conv,
            model_name,
            worker_addr,
            prompt,
            temperature,
            repetition_penalty,
            top_p,
            max_new_tokens,
            images,
        )
    else:
        if use_recommended_config:
            recommended_config = model_api_dict.get("recommended_config", None)
            if recommended_config is not None:
                temperature = recommended_config.get("temperature", temperature)
                top_p = recommended_config.get("top_p", top_p)
                max_new_tokens = recommended_config.get(
                    "max_new_tokens", max_new_tokens
                )

        stream_iter = get_api_provider_stream_iter(
            conv,
            model_name,
            model_api_dict,
            temperature,
            top_p,
            max_new_tokens,
            state,
        )

    html_code = ' <span class="cursor"></span> '
    conv.update_last_message(html_code)
    # Keep flag button invisible during streaming
    yield (state, state.to_gradio_chatbot()) + (
        disable_btn,
        disable_btn,
        invisible_flag_btn,
        disable_btn,
        disable_btn,
    )

    try:
        data = {"text": ""}
        for i, data in enumerate(stream_iter):
            if data["error_code"] == 0:
                output = data["text"].strip()
                conv.update_last_message(output + html_code)
                # Keep flag button invisible during streaming
                yield (state, state.to_gradio_chatbot()) + (
                    disable_btn,
                    disable_btn,
                    invisible_flag_btn,
                    disable_btn,
                    disable_btn,
                )
            else:
                output = data["text"] + f"\n\n(error_code: {data['error_code']})"
                conv.update_last_message(output)
                yield (state, state.to_gradio_chatbot()) + (
                    disable_btn,
                    disable_btn,
                    invisible_flag_btn,  # Keep flag button invisible
                    enable_btn,
                    enable_btn,
                )
                return
        output = data["text"].strip()
        conv.update_last_message(output)
        # Keep flag button invisible at the end
        yield (state, state.to_gradio_chatbot()) + (
            enable_btn,
            enable_btn,
            invisible_flag_btn,
            enable_btn,
            enable_btn,
        )
    except requests.exceptions.RequestException as e:
        conv.update_last_message(
            f"{SERVER_ERROR_MSG}\n\n(error_code: {ErrorCode.GRADIO_REQUEST_ERROR}, {e})"
        )
        yield (state, state.to_gradio_chatbot()) + (
            disable_btn,
            disable_btn,
            invisible_flag_btn,  # Keep flag button invisible
            enable_btn,
            enable_btn,
        )
        return
    except Exception as e:
        conv.update_last_message(
            f"{SERVER_ERROR_MSG}\n\n"
            f"(error_code: {ErrorCode.GRADIO_STREAM_UNKNOWN_ERROR}, {e})"
        )
        yield (state, state.to_gradio_chatbot()) + (
            disable_btn,
            disable_btn,
            invisible_flag_btn,  # Keep flag button invisible
            enable_btn,
            enable_btn,
        )
        return

    finish_tstamp = time.time()
    logger.info(f"eval output: {output}")

    conv.save_new_images(
        has_csam_images=state.has_csam_image, use_remote_storage=use_remote_storage_eval
    )

    filename = get_conv_log_filename(is_vision=state.is_vision)
    # Add eval prefix to log files
    filename = filename.replace("conv.json", "eval-conv.json")

    with open(filename, "a") as fout:
        data = {
            "tstamp": round(finish_tstamp, 4),
            "type": "chat",
            "model": model_name,
            "gen_params": {
                "temperature": temperature,
                "top_p": top_p,
                "max_new_tokens": max_new_tokens,
            },
            "start": round(start_tstamp, 4),
            "finish": round(finish_tstamp, 4),
            "state": state.dict(),
            "ip": get_ip(request),
            "evaluation_mode": True,
        }
        fout.write(json.dumps(data) + "\n")
    get_remote_logger().log(data)


def build_single_model_ui_eval(models, add_promotion_links=False, mode="autoclear"):
    """Build single model UI specifically for evaluation purposes
    Args:
        models: List of available models
        add_promotion_links: Whether to add promotion links (not used in eval)
        mode: "autoclear" for auto-clearing after vote, "normal" for regular behavior
    """

    if mode == "autoclear":
        notice_markdown = """
# 🔬 Direct Chat - Enfore Clear
*Single-turn evaluation mode: Conversation will auto-clear after each vote*
"""
    else:
        notice_markdown = """
# 💬 Direct Chat - Normal
*Multi-turn evaluation mode: Conversation persists until manually cleared*
"""

    state = gr.State()
    gr.Markdown(notice_markdown, elem_id="notice_markdown")

    # Evaluator information section
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Evaluator Information", elem_id="eval_info_label")

    with gr.Row():
        with gr.Column(scale=1):
            textbox_username = gr.Textbox(
                label="Evaluator Name",
                placeholder="Enter your name",
                elem_id="input_box_username",
            )
        with gr.Column(scale=1):
            textbox_question_id = gr.Textbox(
                label="Question ID",
                placeholder="Enter evaluation question ID",
                elem_id="input_box_question_id",
            )

    with gr.Group(elem_id="share-region-eval"):
        with gr.Row(elem_id="model_selector_row"):
            model_selector = gr.Dropdown(
                choices=models,
                value=models[0] if len(models) > 0 else "",
                interactive=True,
                show_label=False,
                container=False,
            )
        with gr.Row():
            with gr.Accordion(
                f"🔍 Model Information ({len(models)} models available)",
                open=False,
            ):
                model_description_md = get_model_description_md(models)
                gr.Markdown(model_description_md, elem_id="model_description_markdown")

        chatbot = gr.Chatbot(
            elem_id="chatbot",
            label="Evaluation Chat Interface",
            height=550,
            show_copy_button=True,
            allow_tags=True,
        )

    with gr.Row():
        textbox = gr.Textbox(
            show_label=False,
            placeholder="👉 Enter your test prompt and press ENTER",
            elem_id="input_box",
        )
        send_btn = gr.Button(value="Send", variant="primary", scale=0)

    # Enhanced feedback section for evaluation
    with gr.Row():
        textbox_feedback = gr.Textbox(
            label="Evaluation Feedback (Required)",
            placeholder="Provide detailed feedback about the model's response quality, accuracy, relevance, etc.",
            elem_id="input_box_feedback",
            visible=False,
            interactive=True,
            lines=1,
        )

    # NEW: Add evaluation checkboxes (Updated to 13 checkboxes)
    with gr.Row():
        with gr.Column():
            checkbox_label = gr.Markdown(
                "### Evaluation Issues (Select all that apply):",
                visible=False,
                elem_id="checkbox_label",
            )

            # Create checkboxes for evaluation criteria (13 total)
            checkbox_factually_incorrect = gr.Checkbox(
                label="Factually Incorrect",
                visible=False,
                elem_id="cb_factually_incorrect",
            )
            checkbox_no_answer = gr.Checkbox(
                label="No answer", visible=False, elem_id="cb_no_answer"
            )
            checkbox_outdated_info = gr.Checkbox(
                label="Outdated info", visible=False, elem_id="cb_outdated_info"
            )
            checkbox_hallucination = gr.Checkbox(
                label="Hallucination", visible=False, elem_id="cb_hallucination"
            )
            checkbox_prompt_misunderstood = gr.Checkbox(
                label="Prompt Misunderstood",
                visible=False,
                elem_id="cb_prompt_misunderstood",
            )
            checkbox_off_topic = gr.Checkbox(
                label="Off-Topic Info", visible=False, elem_id="cb_off_topic"
            )
            checkbox_greeting_ignored = gr.Checkbox(
                label="Greeting ignored (no down-vote)",
                visible=False,
                elem_id="cb_greeting_ignored",
            )
            checkbox_repetitive = gr.Checkbox(
                label="Repetitive", visible=False, elem_id="cb_repetitive"
            )
            checkbox_contradiction = gr.Checkbox(
                label="Contradiction", visible=False, elem_id="cb_contradiction"
            )
            checkbox_code_switching = gr.Checkbox(
                label="Random Code-Switching",
                visible=False,
                elem_id="cb_code_switching",
            )
            checkbox_language_misalignment = gr.Checkbox(
                label="Language misalignment",
                visible=False,
                elem_id="cb_language_misalignment",
            )
            checkbox_verbosity = gr.Checkbox(
                label="Verbosity", visible=False, elem_id="cb_verbosity"
            )
            checkbox_brief_answer = gr.Checkbox(
                label="Brief Answer", visible=False, elem_id="cb_brief_answer"
            )

    with gr.Row() as button_row:
        upvote_btn = gr.Button(value="👍  Upvote", interactive=False)
        downvote_btn = gr.Button(value="👎  Downvote", interactive=False)
        flag_btn = gr.Button(value="⚠️  Problematic", interactive=False, visible=False)
        regenerate_btn = gr.Button(value="🔄  Regenerate", interactive=False)
        clear_btn = gr.Button(value="🗑️  Clear Session", interactive=False)

    with gr.Accordion("Model Parameters", open=False) as parameter_row:
        sys_msg_textbox = gr.Textbox(
            label="System Message",
            placeholder="👉 Enter system message for the model",
            elem_id="sys_msg_input_box",
        )
        temperature = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=0.6,
            step=0.1,
            interactive=True,
            label="Temperature",
        )
        top_p = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=0.98,
            step=0.1,
            interactive=True,
            label="Top P",
        )
        max_output_tokens = gr.Slider(
            minimum=16,
            maximum=4096,
            value=2048,
            step=128,
            interactive=True,
            label="Max output tokens",
        )

    # Register listeners with evaluation-specific functions
    imagebox = gr.State(None)
    btn_list = [upvote_btn, downvote_btn, flag_btn, regenerate_btn, clear_btn]

    # Create list of all checkboxes (Updated to 13 checkboxes)
    checkbox_list = [
        checkbox_factually_incorrect,
        checkbox_no_answer,
        checkbox_outdated_info,
        checkbox_hallucination,
        checkbox_prompt_misunderstood,
        checkbox_off_topic,
        checkbox_greeting_ignored,
        checkbox_repetitive,
        checkbox_contradiction,
        checkbox_code_switching,
        checkbox_language_misalignment,
        checkbox_verbosity,
        checkbox_brief_answer,
    ]

    # Choose vote functions based on mode
    if mode == "autoclear":
        upvote_fn = upvote_last_response_eval_autoclear
        downvote_fn = downvote_last_response_eval_autoclear
        flag_fn = flag_last_response_eval_autoclear
        # For autoclear mode, return all UI elements including state and chatbot
        vote_outputs = [
            state,
            chatbot,
            textbox,
            imagebox,
            upvote_btn,
            downvote_btn,
            flag_btn,
            regenerate_btn,
            clear_btn,
            textbox_feedback,
            textbox_feedback,
            checkbox_label,
        ] + checkbox_list
    else:
        upvote_fn = upvote_last_response_eval
        downvote_fn = downvote_last_response_eval
        flag_fn = flag_last_response_eval
        # For normal mode, return only button states and feedback elements
        vote_outputs = [
            textbox,
            upvote_btn,
            downvote_btn,
            flag_btn,
            textbox_feedback,
            textbox_feedback,
            checkbox_label,
        ] + checkbox_list

    # Updated click handlers to include checkboxes
    upvote_btn.click(
        upvote_fn,
        [
            state,
            model_selector,
            textbox_feedback,
            textbox_username,
            textbox_question_id,
        ]
        + checkbox_list,
        vote_outputs,
    )

    downvote_btn.click(
        downvote_fn,
        [
            state,
            model_selector,
            textbox_feedback,
            textbox_username,
            textbox_question_id,
        ]
        + checkbox_list,
        vote_outputs,
    )

    flag_btn.click(
        flag_fn,
        [
            state,
            model_selector,
            textbox_feedback,
            textbox_username,
            textbox_question_id,
        ]
        + checkbox_list,
        vote_outputs,
    )

    regenerate_btn.click(
        regenerate_eval, state, [state, chatbot, textbox, imagebox] + btn_list
    ).then(
        bot_response_eval,
        [state, temperature, top_p, max_output_tokens],
        [state, chatbot] + btn_list,
    )

    clear_btn.click(
        clear_history_eval,
        None,
        [state, chatbot, textbox, imagebox]
        + btn_list
        + [textbox_feedback, textbox_feedback, checkbox_label]
        + checkbox_list,
    )

    model_selector.change(
        clear_history_eval,
        None,
        [state, chatbot, textbox, imagebox]
        + btn_list
        + [textbox_feedback, textbox_feedback, checkbox_label]
        + checkbox_list,
    )

    textbox.submit(
        add_text_eval,
        [state, model_selector, textbox, imagebox, sys_msg_textbox],
        [state, chatbot, textbox, imagebox] + btn_list,
    ).then(
        bot_response_eval,
        [state, temperature, top_p, max_output_tokens],
        [state, chatbot] + btn_list,
    ).then(
        flash_buttons_eval,
        [],
        btn_list + [textbox_feedback, checkbox_label] + checkbox_list,
    )

    send_btn.click(
        add_text_eval,
        [state, model_selector, textbox, imagebox, sys_msg_textbox],
        [state, chatbot, textbox, imagebox] + btn_list,
    ).then(
        bot_response_eval,
        [state, temperature, top_p, max_output_tokens],
        [state, chatbot] + btn_list,
    ).then(
        flash_buttons_eval,
        [],
        btn_list + [textbox_feedback, checkbox_label] + checkbox_list,
    )

    return [state, model_selector]
