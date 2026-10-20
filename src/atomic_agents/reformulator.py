import copy

from src.models import MessageRole, Model, ChatMessage
from src.logger import logger


async def prepare_response(original_task: str, inner_messages, reformulation_model: Model) -> str:
    messages = [
        {
            "role": MessageRole.SYSTEM,
            "content": [
                {
                    "type": "text",
                    "text": f"""Earlier you were asked the following:

{original_task}

Your team then worked diligently to address that request. Read below a transcript of that conversation:""",
                }
            ],
        }
    ]

    # copy them to this context
    try:
        for message in inner_messages:
            # Convert ChatMessage objects to dicts
            if isinstance(message, ChatMessage):
                message_dict = message.dict()
            else:
                message_dict = message
            
            if not message_dict.get("content"):
                continue
            
            message_dict = copy.deepcopy(message_dict)
            message_dict["role"] = MessageRole.USER
            # Normalize string content to the list-of-parts format
            if isinstance(message_dict.get("content"), str):
                message_dict["content"] = [{"type": "text", "text": message_dict["content"]}]
            messages.append(message_dict)
    except Exception:
        messages += [{"role": MessageRole.ASSISTANT, "content": [{"type": "text", "text": str(inner_messages)}]}]

    # ask for the final answer
    messages.append(
        {
            "role": MessageRole.USER,
            "content": [
                {
                    "type": "text",
                    "text": f"""
Read the above conversation and output a FINAL ANSWER to the question. The question is repeated here for convenience:

{original_task}

To output the final answer, use the following template: FINAL ANSWER: [YOUR FINAL ANSWER]
Your FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings.
ADDITIONALLY, your FINAL ANSWER MUST adhere to any formatting instructions specified in the original question (e.g., alphabetization, sequencing, units, rounding, decimal places, etc.)
* You MUST pay attention to the required units of the calculation result. For example, if the question asks "how many thousand hours...", then the answer `1000 hours` should be `1`, not `1000`.
* You MUST pay attention to extracting key stage names, personal names, and location names when the task required.
* If you are asked for a number, express it numerically (i.e., with digits rather than words), don't use commas, and DO NOT INCLUDE UNITS such as $ or USD or percent signs unless specified otherwise.
* If you are asked for a string, don't use articles or abbreviations (e.g. for cities), unless specified otherwise. Don't output any final sentence punctuation such as '.', '!', or '?'.
* If you are asked for a comma separated list, apply the above rules depending on whether the elements are numbers or strings.
* If you are unable to determine the final answer, output 'FINAL ANSWER: Unable to determine'
""",
                }
            ],
        }
    )

    messages = [
        ChatMessage.from_dict(msg) for msg in messages
    ]

    response = await reformulation_model(messages)
    response = response.content

    final_answer = response.split("FINAL ANSWER: ")[-1].strip()
    logger.info(f"> Reformulated answer: {final_answer}")

    return final_answer
