# from langchain_community.chat_models import ChatOllama
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
from typing import List, Union

def chat_with_feedback(report: str, chat_history: List[Union[HumanMessage, AIMessage]], user_input: str) -> tuple:
    """
    Processes user input with context from exercise feedback report and chat history to generate appropriate responses.
    
    Args:
        report (str): Exercise feedback report containing analysis of user's form
        chat_history (List): List of previous chat messages between user and bot
        user_input (str): Current user message/question
        
    Returns:
        tuple: (response, updated_chat_history)
    """
    # Initialize the LLM
    llm = ChatOllama(model="llama2", base_url="http://localhost:11434")
    
    # Create a context-rich prompt combining report and user input
    system_context = f"""You are a knowledgeable fitness assistant. Use the following exercise report as context for answering the user's questions. Keep your responses focused on the user's specific situation and previous feedback.

    Exercise Report:
    {report}
    """
    
    # Add the system context as the first message if chat history is empty
    if not chat_history:
        chat_history.append(AIMessage(content=system_context))
    
    # Add the new user input to chat history
    chat_history.append(HumanMessage(content=user_input))
    
    # Prepare the full conversation context
    messages = chat_history.copy()
    
    # Generate response using the LLM
    response = llm.invoke(messages, stream=False)

    # Extract only the content from the response
    response_content = response.content
    
    # Add the assistant's response to chat history
    chat_history.append(AIMessage(content=response_content))
    
    return response_content, chat_history

# # Example usage:
# if __name__ == "__main__":
#     # Initialize empty chat history
#     chat_history = []
    
#     # Example report
#     example_report = """당신의 스쿼트 자세는 전반적으로 우수합니다! 척추의 중립이 잘 지켜지고 있고 무릎을 구부렸을 때 무릎의 가장 앞쪽 부분이 발가락을 넘어가지 않습니다다. 다만 reps가 진행됨에 따라 등이 살짝 굽어지는 경향이 있습니다. 등이 굽어지지 않도록 시선을 정면보다 살짝 위쪽에 두면 완벽한 자세가 될 것 같습니다!"""
    
#     # Example user input
#     example_input = """고마워. 그런데 내가 무릎이 좋지 않아 무릎을 90도가 되도록 굽히기 어려워 대신 미니 스쿼트를 하고 싶은데 주의할 점을 알려줘."""
    
#     # Get response and updated history
#     response, updated_history = chat_with_feedback(example_report, chat_history, example_input)

#     print(response)