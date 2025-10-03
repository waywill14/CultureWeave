import from config { setup_api_keys, validate_api_keys }
import os;
import from ai_functions { 
    generate_ai_story, provide_cultural_insights, ai_translate_story,
    verify_cultural_authenticity, generate_ar_experience, generate_branching_story,
    generate_cultural_music, generate_audio_narration, generate_cultural_illustration,
    process_cultural_artifact, apply_cultural_filters
}
import from nodes { 
    story_node, translation_node, knowledge_node, verification_node,
    immersive_node, audio_node, ar_node, archive_node, elder_node, sacred_node
}
import from walkers { StoryWeaver, KnowledgeKeeper }
import implementations;

# Interactive CultureWeave Application
def show_welcome() {
    print("=" * 60);
    print("🌟 WELCOME TO VORTEX-AI-CULTUREWEAVE 🌟");
    print("GenAI for Indigenous Knowledge Preservation");
    print("=" * 60);
    print("📱 Features: Real AI Stories, AR Experiences, Audio Narration");
    print("🌍 Supporting 42+ Kenyan ethnic groups and 2,000+ African languages");
    print("🚀 Powered by: Groq Llama 3.1 8B Instant API");
    print("=" * 60);
}

def select_language(title: str) -> str {
    print("\n" + title);
    print("1. English");
    print("2. Kiswahili");
    print("3. Kikuyu");
    print("4. Luo");
    print("5. Kamba");
    print("6. Maasai");
    print("7. Other (type any language)");
    languages = ["English", "Kiswahili", "Kikuyu", "Luo", "Kamba", "Maasai", "Other"];
    choice = get_user_choice(languages);
    if choice == 7 {
        custom_lang = get_user_input("Enter language name (any language supported):");
        if not custom_lang {
            return "English";
        }
        return custom_lang;
    }
    return languages[choice - 1];
}

def show_menu() {
    print("\n🎯 What would you like to do?");
    print("1. 📖 Generate AI-Powered Cultural Story");
    print("2. 🏛️ Preserve Cultural Knowledge");
    print("3. 🔍 Verify Cultural Authenticity");
    print("4. 🌍 Translate Cultural Content");
    print("5. 🎨 Create Cultural Illustration");
    print("6. 🎵 Generate Cultural Music");
    print("7. 🥽 Create AR Cultural Experience");
    print("8. ❌ Exit");
    print("-" * 40);
}

def get_user_input(prompt: str) -> str {
    print(prompt);
    print(">>> ", end="");
    # Real user input using Python's input function through Jac
    import subprocess;
    result = subprocess.run(["python3", "-c", "print(input())"], capture_output=True, text=True);
    user_input = result.stdout.strip();
    return user_input;
}

def get_user_choice(options: list) -> int {
    while True {
        print("Enter your choice (number 1-" + str(len(options)) + "):");
        print(">>> ", end="");
        import subprocess;
        result = subprocess.run(["python3", "-c", "print(input())"], capture_output=True, text=True);
        choice_str = result.stdout.strip();
        
        try {
            choice = int(choice_str);
            if choice >= 1 and choice <= len(options) {
                return choice;
            } else {
                print("Please enter a number between 1 and " + str(len(options)));
            }
        } except Exception {
            print("Please enter a valid number");
        }
    }
}

def run_interactive_system() {
    print("\n🎯 CULTUREWEAVE - INTERACTIVE SYSTEM");
    print("Welcome! Type your responses to interact with the AI system.");
    
    while True {
        print("\n" + "="*60);
        show_menu();
        
        menu_options = [
            "Generate AI-Powered Cultural Story",
            "Preserve Cultural Knowledge", 
            "Verify Cultural Authenticity",
            "Translate Cultural Content",
            "Create Cultural Illustration",
            "Generate Cultural Music",
            "Create AR Cultural Experience",
            "Exit"
        ];
        
        choice = get_user_choice(menu_options);
        
        if choice == 1 {
            handle_story_generation();
        } elif choice == 2 {
            handle_knowledge_preservation();
        } elif choice == 3 {
            handle_cultural_verification();
        } elif choice == 4 {
            handle_translation();
        } elif choice == 5 {
            handle_illustration();
        } elif choice == 6 {
            handle_music_generation();
        } elif choice == 7 {
            handle_ar_experience();
        } elif choice == 8 {
            print("\n👋 Thank you for using CultureWeave!");
            print("🌟 Preserving indigenous knowledge through AI.");
            break;
        }
        
        # Ask if user wants to continue
        print("\n" + "="*60);
        continue_choice = get_user_input("Press Enter to return to main menu, or type 'exit' to quit:");
        if continue_choice.lower() == "exit" {
            print("\n👋 Thank you for using CultureWeave!");
            print("🌟 Preserving indigenous knowledge through AI.");
            break;
        }
    }
}

def handle_story_generation() {
    print("\n📖 AI-POWERED CULTURAL STORY GENERATOR");
    print("-" * 40);
    
    # Get user inputs
    user_prompt = get_user_input("Enter your story prompt (e.g., 'A tale about courage in my village'):");
    
    print("\nChoose story type:");
    print("1. folklore");
    print("2. legend"); 
    print("3. myth");
    print("4. fable");
    story_types = ["folklore", "legend", "myth", "fable"];
    story_choice = get_user_choice(story_types);
    story_type = story_types[story_choice - 1];
    
    language = select_language("Choose language:");
    
    print(f"\n🤖 Generating story with prompt: '{user_prompt}'");
    print(f"📝 Story type: {story_type}");
    print(f"🌍 Language: {language}");
    print("\n⏳ Please wait while AI generates your cultural story...\n");
    
    # Generate AI story using user's input
    generated_story = generate_ai_story(user_prompt, story_type, language);
    
    print("✅ STORY GENERATED:");
    print("=" * 50);
    print(generated_story);
    print("=" * 50);
    
    # Generate additional cultural content
    print("\n🎨 Generating cultural illustration...");
    illustration = generate_cultural_illustration(generated_story, "traditional_art", f"{language} cultural storytelling");
    print(f"🖼️ Illustration: {illustration}");
    
    print("\n🎵 Generating cultural music...");
    music = generate_cultural_music("traditional", ["drums", "voice"], "storytelling");
    print(f"🎶 Music: {music}");
    
    print("\n🎙️ Generating audio narration...");
    narration = generate_audio_narration(generated_story, language, "elder");
    print(f"🔊 Narration: {narration}");
}

def handle_knowledge_preservation() {
    print("\n🏛️ CULTURAL KNOWLEDGE PRESERVATION");
    print("-" * 40);
    
    cultural_element = get_user_input("Enter cultural element to preserve (e.g., 'Traditional pottery techniques'):");
    
    cultural_group = get_user_input("Enter cultural group (e.g., 'Kikuyu', 'Maasai', 'Luo'):");
    
    print("\nChoose privacy level:");
    print("1. public - Accessible to everyone");
    print("2. community - Accessible to community members only");
    print("3. sacred - Restricted access with elder approval");
    privacy_levels = ["public", "community", "sacred"];
    privacy_choice = get_user_choice(privacy_levels);
    privacy_level = privacy_levels[privacy_choice - 1];
    
    print(f"\n📚 Preserving: '{cultural_element}'");
    print(f"👥 Cultural group: {cultural_group}");
    print(f"🔒 Privacy level: {privacy_level}");
    print("\n⏳ Processing cultural knowledge preservation...\n");
    
    # Create and spawn KnowledgeKeeper
    root spawn KnowledgeKeeper(
        cultural_element,
        "pending",
        cultural_group,
        privacy_level,
        False
    );
}

def handle_cultural_verification() {
    print("\n🔍 CULTURAL AUTHENTICITY VERIFICATION");
    print("-" * 40);
    
    element = get_user_input("Enter cultural element to verify (e.g., 'Maasai jumping dance'):");
    
    context = get_user_input("Enter cultural context (e.g., 'Traditional initiation ceremonies'):");
    
    print(f"\n🔍 Verifying: '{element}'");
    print(f"📋 Context: {context}");
    print("\n⏳ AI is verifying cultural authenticity...\n");
    
    # Generate verification
    verification_result = verify_cultural_authenticity(element, context);
    print(f"✅ Verification Result: {verification_result}");
}

def handle_translation() {
    print("\n🌍 CULTURAL CONTENT TRANSLATION");
    print("-" * 40);
    
    story = get_user_input("Enter text to translate:");
    
    target_language = select_language("Choose target language:");
    
    cultural_context = get_user_input("Enter cultural context (e.g., 'Community cooperation'):");
    
    print(f"\n🌍 Translating: '{story}'");
    print(f"🎯 Target language: {target_language}");
    print(f"📋 Context: {cultural_context}");
    print("\n⏳ AI is translating with cultural sensitivity...\n");
    
    # Generate translation
    translation = ai_translate_story(story, target_language, cultural_context);
    print(f"✅ Translation: {translation}");
}

def handle_illustration() {
    print("\n🎨 CULTURAL ILLUSTRATION GENERATOR");
    print("-" * 40);
    
    description = get_user_input("Describe what you want illustrated:");
    
    print("\nChoose art style:");
    print("1. traditional_art - Classic cultural art style");
    print("2. modern - Contemporary interpretation");
    print("3. abstract - Artistic abstraction");
    styles = ["traditional_art", "modern", "abstract"];
    style_choice = get_user_choice(styles);
    style = styles[style_choice - 1];
    
    cultural_context = get_user_input("Enter cultural context:");
    
    print(f"\n🎨 Creating illustration: '{description}'");
    print(f"🖼️ Style: {style}");
    print(f"📋 Context: {cultural_context}");
    print("\n⏳ AI is generating cultural illustration...\n");
    
    # Generate illustration
    illustration = generate_cultural_illustration(description, style, cultural_context);
    print(f"✅ Illustration: {illustration}");
}

def handle_music_generation() {
    print("\n🎵 CULTURAL MUSIC GENERATOR");
    print("-" * 40);
    
    print("\nChoose music genre:");
    print("1. traditional - Classic cultural music");
    print("2. fusion - Modern blend with traditional");
    print("3. ceremonial - Sacred and ritual music");
    genres = ["traditional", "fusion", "ceremonial"];
    genre_choice = get_user_choice(genres);
    genre = genres[genre_choice - 1];
    
    print("\nSelect instruments (you can choose multiple):");
    print("Available: drums, flute, voice, harp, guitar, violin");
    instruments_input = get_user_input("Enter instruments separated by commas:");
    instruments = [inst.strip() for inst in instruments_input.split(",")];
    
    print("\nChoose mood:");
    print("1. celebratory - Joyful and festive");
    print("2. ceremonial - Sacred and solemn");
    print("3. peaceful - Calm and meditative");
    print("4. energetic - Lively and dynamic");
    moods = ["celebratory", "ceremonial", "peaceful", "energetic"];
    mood_choice = get_user_choice(moods);
    mood = moods[mood_choice - 1];
    
    print(f"\n🎵 Generating music:");
    print(f"🎼 Genre: {genre}");
    print(f"🎹 Instruments: {', '.join(instruments)}");
    print(f"😊 Mood: {mood}");
    print("\n⏳ AI is composing cultural music...\n");
    
    # Generate music
    music = generate_cultural_music(genre, instruments, mood);
    print(f"✅ Music: {music}");
}

def handle_ar_experience() {
    print("\n🥽 AR CULTURAL EXPERIENCE CREATOR");
    print("-" * 40);
    
    print("\nChoose cultural site:");
    print("1. sacred_ground - Holy and spiritual places");
    print("2. village - Traditional community settings");
    print("3. market - Cultural trading centers");
    print("4. ceremony - Ritual and celebration spaces");
    sites = ["sacred_ground", "village", "market", "ceremony"];
    site_choice = get_user_choice(sites);
    site = sites[site_choice - 1];
    
    cultural_context = get_user_input("Enter cultural context (e.g., 'Ancient rituals and traditions'):");
    
    language = select_language("Choose language for narration:");
    
    print(f"\n🥽 Creating AR experience:");
    print(f"📍 Site: {site}");
    print(f"📋 Context: {cultural_context}");
    print(f"🌍 Language: {language}");
    print("\n⏳ AI is creating immersive AR experience...\n");
    
    # Generate AR experience
    ar_experience = generate_ar_experience(site, cultural_context, language);
    print(f"✅ AR Experience: {ar_experience}");
}

# Main entry point with interactive menu
with entry:__main__ {
    # Configure API keys
    groq_key = os.environ.get("GROQ_API_KEY", "");
    serper_key = os.environ.get("SERPER_API_KEY", "");
    setup_api_keys(groq_key, serper_key);
    
    # Validate API keys before proceeding
    if not validate_api_keys() {
        print("❌ API key validation failed. Exiting...");
        exit;
    }
    
    # Show welcome message
    show_welcome();
    
    # Start interactive system
    print("\n🎮 Ready to begin your CultureWeave experience!");
    start_input = get_user_input("Press Enter to start:");
    
    # Run interactive system
    run_interactive_system();
}
