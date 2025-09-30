import os;

# Configuration for API keys and settings
def setup_api_keys(groq_key: str, serper_key: str) {
    # Set Groq API key
    os.environ["GROQ_API_KEY"] = groq_key;
    
    # Set Serper API key for search functionality
    os.environ["SERPER_API_KEY"] = serper_key;
    
    print("✅ API keys configured successfully");
    print("🚀 Groq API key set (Llama 3.1 8B Instant)");
    print("🔍 Serper API key set");
}

def get_api_keys() -> dict {
    return {
        "groq": os.environ.get("GROQ_API_KEY", ""),
        "serper": os.environ.get("SERPER_API_KEY", "")
    };
}

def validate_api_keys() -> bool {
    keys = get_api_keys();
    
    if not keys["groq"] {
        print("❌ Groq API key not found");
        return False;
    }
    
    if not keys["serper"] {
        print("❌ Serper API key not found");
        return False;
    }
    
    print("✅ All API keys validated");
    return True;
}
