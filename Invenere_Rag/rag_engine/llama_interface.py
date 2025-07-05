import subprocess

def query_llama(prompt: str, model: str = "llama3") -> str:
    try:
        print("Calling ollama with prompt:", repr(prompt))
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt,
            text=True,
            capture_output=True
        )
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        print("RETURN CODE:", result.returncode)
        return result.stdout.strip()
    except Exception as e:
        print("Error querying LLaMA:", e)
        return f"❌ Error querying LLaMA: {e}"
