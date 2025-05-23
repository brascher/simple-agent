import boto3
import json

# from langchain.tools import tool
# from  langchain_core.messages import HumanMessage
# from langchain.prebuilt import create_react_agent


def main():
  bedrock_client = boto3.client("bedrock-runtime")
  model_id = "amazon.nova-lite-v1:0"
  inf_params = {
    "maxTokens": 2048,
    "temperature": 0.1,
    "topP": 0.9,
    "topK": 20
  }

  print("Welcome! I'm your AI assistant. Type '/quit' to exit.")
  print("You can ask me to perform calcuations or chat with me.")

  while True:
    user_input = input("You: ").strip()
    
    if user_input.lower() == "/quit":
      print("Goodbye!")
      break

    response = bedrock_client.invoke_model_with_response_stream(
      modelId=model_id,
      body=json.dumps({
        "schemaVersion": "messages-v1",
        "inferenceConfig": inf_params,
        "messages": [
          {
            "role": "user",
            "content": [
              {
                "text": user_input
              }
            ]
          }
        ]
      })
    )

    request_id = response.get("ResponseMetadata", {}).get("RequestId")
    print(f"Request ID: {request_id}")
    print("Awaiting first token ...")

    result = response.get("body")
    if result:
      print("\nAssistant: ", end="")
      for event in result:
        chunk = event.get("chunk")
        if chunk:
          chunk_json = json.loads(chunk.get("bytes").decode("utf-8"))
          content_blk_delta = chunk_json.get("contentBlockDelta")
          if content_blk_delta:
            print(content_blk_delta.get("delta").get("text"), end="")


if __name__ == "__main__":
  main()
