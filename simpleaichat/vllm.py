from typing import Any, Dict, List, Set, Union

import orjson
from httpx import AsyncClient, Client
from pydantic import HttpUrl

from .models import ChatMessage, ChatSession
from .utils import remove_a_key

tool_prompt = """From the list of tools below:
- Reply ONLY with the number of the tool appropriate in response to the user's last message.
- If no tool is appropriate, ONLY reply with \"0\".

{tools}"""


class vLLMSession(ChatSession):
    api_url: HttpUrl = "https://localhost:8000/v1/chat/completions" or "https://localhost:8080/v1/chat/completions"
    input_fields: Set[str] = {"role", "content", "name"}
    system: str = "You are a helpful assistant."
    params: Dict[str, Any] = {"temperature": 0.3}

    def prepare_request(
        self,
        prompt: str,
        system: str = None,
        params: Dict[str, Any] = None,
        stream: bool = False,
        input_schema: Any = None,
        output_schema: Any = None,
    ):
        headers = {
            "Content-Type": "application/json",
            # vLLM doesn't need an API Key
            # "Authorization": f"Bearer {self.auth['api_key'].get_secret_value()}",
        }

        system_message = ChatMessage(role="system", content=system or self.system)

        # Handle regular prompts and "function calls" with guided_json
        if input_schema:
            assert isinstance(prompt, input_schema), f"prompt must be an instance of {input_schema.__name__}"
            user_message = ChatMessage(
                role="user",
                content=prompt.model_dump_json(),
                name=input_schema.__name__,
            )
        else:
            user_message = ChatMessage(role="user", content=prompt)


        gen_params = params or self.params
        
        # vLLM doesn't natively support function calling, but it does support guided json
        # We will use guided json as a stand in for function calling
        if input_schema or output_schema:
            gen_params["guided_json"] = self.schema_to_guided_json(input_schema, output_schema)

        data = {
            "model": self.model,
            "messages": self.format_input_messages(system_message, user_message),
            "stream": stream,
            **gen_params,
        }

        return headers, data, user_message

    def schema_to_guided_json(self, input_schema, output_schema):
        schemas = {}
        if input_schema:
            schemas["input"] = input_schema.schema()
        if output_schema:
            schemas["output"] = output_schema.schema()
        return schemas