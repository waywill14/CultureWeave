import subprocess;

def test_real_input() {
    print("Testing real user input in Jac...");
    print("What is your name?");
    print(">>> ", end="");
    
    result = subprocess.run(["python3", "-c", "print(input())"], capture_output=True, text=True);
    user_name = result.stdout.strip();
    
    print(f"Hello, {user_name}! The input system works.");
}

with entry:__main__ {
    test_real_input();
}
