from src.agent.graph import app

def main():
    inputs = {
        "question": "What are the types of agent memory?"
    }    
    response =  app.invoke(inputs)
    print(response.get("generation", "No found"))

if __name__ == "__main__":
    main()
