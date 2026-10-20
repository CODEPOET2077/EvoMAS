from typing import (
    Any,
    Optional,
    List,
)
import asyncio

from src.atomic_agents.general_agent import GeneralAgent
from src.models import ChatMessage, MessageRole
from src.registry import AGENT
from src.logger import logger, LogLevel


@AGENT.register_module(name="multi_generate_agent", force=True)
class MultiGenerateAgent(GeneralAgent):
    """
    MultiGenerateAgent generates multiple responses for the same task.
    
    Similar to MultiGenerateCoT:
    - Takes task_description as input
    - Generates 3 responses in parallel from the same prompt
    - Returns {"response": [response1, response2, response3]}
    
    Implementation:
    - Reuses GeneralAgent's initialization entirely
    - Only overrides run() to implement multi-generation
    - Uses its own prompt template: multi_generate_agent.yaml
    """
    
    # __init__ is inherited from GeneralAgent; the prompt template
    # (multi_generate_agent.yaml) is set via template_path in the config.
    
    async def run(
        self,
        task: str,
        stream: bool = False,
        reset: bool = True,
        images: Optional[List[Any]] = None,
        layer_memory: Optional[str] = None,
        additional_args: Optional[dict] = None,
        max_steps: Optional[int] = None,
    ):
        """
        Run MultiGenerateAgent and generate 3 responses.
        
        Reuses GeneralAgent's prompt construction, then calls the model 3 times directly.
        
        Args:
            task: Task description.
            stream: Whether to stream output (not supported yet).
            reset: Whether to reset memory.
            images: List of images.
            layer_memory: Memory from previous layers.
            additional_args: Extra arguments.
            max_steps: Maximum number of steps (unused).
            
        Returns:
            dict: {"response": [response1, response2, response3]}
        """
        # 1. Set task and layer_memory
        self.task = task
        self.layer_memory = layer_memory
        
        # 2. Log task start
        self.logger.log_task(
            content=task.strip(),
            subtitle=f"{type(self.model).__name__} - {(self.model.model_id if hasattr(self.model, 'model_id') else '')}",
            level=LogLevel.INFO,
            title=self.name if hasattr(self, "name") else "MultiGenerateAgent",
        )
        
        # 3. Reuse the parent's prompt builders.
        # GeneralAgent.initialize_task_instruction() reads self.task, so it must be set first.
        task_instruction = self.initialize_task_instruction()  # fills {{task}}
        system_prompt = self.initialize_system_prompt()
        user_prompt = self.initialize_user_prompt()
        
        # 4. Build messages (same prompt structure as GeneralAgent)
        messages = self._build_messages(
            system_prompt=system_prompt,
            task_instruction=task_instruction,
            user_prompt=user_prompt,
            images=images,
            layer_memory=layer_memory
        )
        
        # 5. Log generation start
        self.logger.log_rule("Generating 3 responses in parallel", level=LogLevel.INFO)
        logger.info(f"[MultiGenerateAgent] Starting to generate 3 responses for task: {task[:100]}...")
        
        # 6. Generate 3 responses in parallel
        try:
            responses = await asyncio.gather(
                self._generate_with_logging(messages, 1),
                self._generate_with_logging(messages, 2),
                self._generate_with_logging(messages, 3),
            )
        except Exception as e:
            logger.error(f"Error generating responses in MultiGenerateAgent: {e}")
            # On failure, return an error message for each response
            error_response = ChatMessage(
                role=MessageRole.ASSISTANT,
                content=f"Error in generating responses: {str(e)}"
            )
            responses = [error_response, error_response, error_response]
        
        # 7. Extract response contents
        response_contents = [self._extract_content(r) for r in responses]
        
        # 8. Log completion
        self.logger.log_rule("All 3 responses generated", level=LogLevel.INFO)
        logger.info(f"[MultiGenerateAgent] Successfully generated 3 responses")
        for i, content in enumerate(response_contents, 1):
            logger.info(f"[MultiGenerateAgent] Response {i} length: {len(content)} characters")
        
        # 9. Return
        return {"response": response_contents}
    
    async def _generate_with_logging(self, messages: List[ChatMessage], response_num: int) -> ChatMessage:
        """
        Generate a single response and log it in the same format as other agents.
        
        Args:
            messages: List of messages.
            response_num: Response index (1, 2, 3).
            
        Returns:
            ChatMessage: The model response.
        """
        logger.info(f"[MultiGenerateAgent] Generating response {response_num}/3...")
        try:
            # Model generation kwargs (e.g. temperature)
            model_kwargs = self._get_model_kwargs()
            response = await self.model(messages, **model_kwargs)
            
            content = response.content if isinstance(response.content, str) else str(response.content)
            if not content and hasattr(response, 'raw'):
                content = str(response.raw)
            
            # Log output in the same format as other agents
            self.logger.log_markdown(
                content=content,
                title=f"Output message of the LLM (Response {response_num}/3):",
                level=LogLevel.DEBUG,
            )
            
            logger.info(f"[MultiGenerateAgent] Response {response_num}/3 completed (length: {len(content)} chars)")
            return response
        except Exception as e:
            logger.error(f"[MultiGenerateAgent] Error generating response {response_num}/3: {e}")
            raise
    
    def _build_messages(
        self,
        system_prompt: str,
        task_instruction: str,
        user_prompt: str,
        images: Optional[List[Any]] = None,
        layer_memory: Optional[str] = None,
    ) -> List[ChatMessage]:
        """
        Build the ChatMessage list using GeneralAgent's prompt structure.
        
        Args:
            system_prompt: System prompt.
            task_instruction: Task instruction.
            user_prompt: User prompt.
            images: List of images.
            layer_memory: Memory from previous layers.
            
        Returns:
            List[ChatMessage]: List of messages.
        """
        messages = []
        
        # 1. System prompt
        messages.append(
            ChatMessage(
                role=MessageRole.SYSTEM,
                content=[{"type": "text", "text": system_prompt}]
            )
        )
        
        # 2. Images, if any
        if images:
            messages[0].content.extend([{"type": "image", "image": img} for img in images])
        
        # 3. User content: layer_memory (if any) + task_instruction + user_prompt
        user_content_parts = []
        
        if layer_memory:
            user_content_parts.append(f"Previous context:\n{layer_memory}\n")
        
        user_content_parts.append(task_instruction)
        user_content_parts.append(f"\n{user_prompt}")
        
        user_content = "\n".join(user_content_parts)
        
        # 4. User prompt
        messages.append(
            ChatMessage(
                role=MessageRole.USER,
                content=[{"type": "text", "text": user_content}]
            )
        )
        
        return messages
    
    def _extract_content(self, chat_message: ChatMessage) -> str:
        """
        Extract text content from a ChatMessage.
        
        Args:
            chat_message: ChatMessage object.
            
        Returns:
            str: Extracted text.
        """
        if not isinstance(chat_message, ChatMessage):
            return str(chat_message)
        
        content = chat_message.content
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            # Multimodal content (text + image)
            text_parts = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                elif isinstance(item, str):
                    text_parts.append(item)
            return "".join(text_parts)
        else:
            return str(content)
