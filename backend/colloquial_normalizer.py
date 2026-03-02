"""
Colloquial Normalizer - LLM-based Turkish colloquial speech normalization

Converts formal written Turkish text to colloquial spoken forms to ensure
transcript-audio alignment in TTS training data.

Strategy B: Fully LLM-based normalization.
"""

import json
import os
import requests
from typing import List, Dict, Optional
from dotenv import load_dotenv

load_dotenv()

# Global settings — read from .env so toggle survives Flask restarts
_colloquial_enabled = os.getenv("COLLOQUIAL_ENABLED", "false").lower() == "true"
_current_provider = os.getenv("LLM_PROVIDER", "openai")
_ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
_ollama_model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

# OpenAI client (lazy initialized)
_openai_client = None


def is_enabled() -> bool:
    """Check if colloquial normalization is enabled."""
    return _colloquial_enabled


def set_enabled(enabled: bool):
    """Enable or disable colloquial normalization. Persists to .env file."""
    global _colloquial_enabled
    _colloquial_enabled = enabled
    
    # Persist to .env so it survives Flask restarts
    _persist_to_env("COLLOQUIAL_ENABLED", "true" if enabled else "false")
    
    print(f"{'✅' if enabled else '❌'} Colloquial normalization: {'ON' if enabled else 'OFF'}")


def _persist_to_env(key: str, value: str):
    """Update a key in the .env file."""
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    
    lines = []
    found = False
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    
    new_lines = []
    for line in lines:
        if line.strip().startswith(f"{key}="):
            new_lines.append(f"{key}={value}\n")
            found = True
        else:
            new_lines.append(line)
    
    if not found:
        new_lines.append(f"{key}={value}\n")
    
    with open(env_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)


def _get_openai_client():
    """Get or initialize OpenAI client."""
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise Exception("OPENAI_API_KEY not set")
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


def _build_normalization_prompt(text: str) -> str:
    """Build the LLM prompt for colloquial normalization."""
    return f"""Sen bir Türkçe dilbilgisi ve konuşma dili uzmanısın. Verilen cümleyi, bir Türk konuşucunun günlük doğal konuşmada söyleyeceği şekle çevir.

KRİTİK DİLBİLGİSİ KURALI — SIFAT-FİİL vs FİİL AYRIMI:
Kısaltma SADECE kelime FİİL olarak (yüklem olarak) kullanıldığında yapılır.
Eğer kelime bir İSİMDEN ÖNCE geliyorsa (sıfat-fiil/ortaç), KESİNLİKLE kısaltma YAPMA.

DOĞRU ÖRNEKLER:
✅ "Yarın oraya gideceğim." → "Yarın oraya gidicem." (fiil, cümle sonu → KISALT)
✅ "Bunu yapacağım." → "Bunu yapıcam." (fiil, cümle sonu → KISALT)
❌ "Gideceğim adresten emin değilim." → "Gideceğim adresten emin değilim." (sıfat-fiil + isim → KISALTMA)
❌ "Alacağım kitabı söyle." → "Alacağım kitabı söyle." (sıfat-fiil + isim → KISALTMA)
❌ "Yapacağım işler çok fazla." → "Yapacağım işler çok fazla." (sıfat-fiil + isim → KISALTMA)

TEST: Eğer kelimenin hemen SONRASINDA bir isim/nesne geliyorsa → KISALTMA
      Eğer kelime cümlenin/cümleciğin SONUNDA veya virgülden/noktadan önceyse → KISALT

DİĞER KURALLAR:
- Kelimelerin anlamını DEĞİŞTİRME
- Cümle yapısını DEĞİŞTİRME (kelime sırasını değiştirme, kelime ekleme/çıkarma yapma)
- Noktalama işaretlerini koru
- Eğer kelime zaten konuşma dilinde ise AYNEN BIRAK

YAYGIN DÖNÜŞÜMLER (sadece fiil pozisyonunda):
- Gelecek zaman fiili: -eceğim → -ecem, -acağım → -acam, -eceğiz → -ecez
- Geçmiş/hikaye: -iyordum → -iyodum, -ıyordun → -ıyodun, -iyorduk → -iyoduk
- Kaynaşmalar: bir şey → bişey
- Yer: burada → burda, orada → orda, nerede → nerde
- Soru: değil mi → dimi

ŞİMDİKİ ZAMAN — ŞAHIS BAZINDA KURALLAR:
Kısaltılabilir (2. tekil, 3. tekil, 3. çoğul):
- -iyorsun → -iyosun (yapıyorsun → yapıyosun)
- -iyor → -iyo (gidiyor → gidiyo)
- -iyorlar → -iyolar (geliyorlar → geliyolar)
- ne yapıyorsun → napıyosun

KISALTMA (1. tekil, 1. çoğul, 2. çoğul):
- istiyorum → istiyorum (DEĞİŞTİRME!)
- gidiyorum → gidiyorum (DEĞİŞTİRME!)
- yapıyoruz → yapıyoruz (DEĞİŞTİRME!)
- biliyorsunuz → biliyorsunuz (DEĞİŞTİRME!)
- -iyorum, -iyoruz, -iyorsunuz eklerini KESİNLİKLE kısaltma!

DAHA FAZLA ÖRNEK:
"Bir şey söyleyeceğim sana." → "Bişey söylicem sana."
"Göreceğim insanlar çok fazla." → "Göreceğim insanlar çok fazla." (sıfat-fiil → DEĞİŞMEZ)
"Onu mutlaka göreceğim." → "Onu mutlaka görecem." (fiil → KISALT)
"Burada bir şey var mı?" → "Burda bişey var mı?"
"Seni çok istiyorum." → "Seni çok istiyorum." (şimdiki zaman → DEĞİŞMEZ)

ZORUNLU FORMAT - Sadece aşağıdaki JSON'u döndür, başka hiçbir şey yazma:
{{"spoken": "normalize edilmiş cümle", "changes": [{{"from": "orijinal kelime", "to": "değişen kelime"}}]}}

Cümle: "{text}" """


def _normalize_with_openai(prompt: str) -> dict:
    """Call OpenAI to normalize text."""
    client = _get_openai_client()

    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {
                "role": "system",
                "content": "Sen bir Türkçe konuşma dili uzmanısın. Yazı dilindeki cümleleri konuşma diline çevirirsin. Sadece JSON döndür."
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,  # Low temperature for consistency
        max_tokens=500
    )

    content = response.choices[0].message.content.strip()
    return _parse_json_result(content)


def _normalize_with_ollama(prompt: str) -> dict:
    """Call Ollama to normalize text."""
    try:
        response = requests.post(
            f"{_ollama_base_url}/api/generate",
            json={
                "model": _ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3
                }
            },
            timeout=60
        )

        if response.status_code != 200:
            raise Exception(f"Ollama API error: {response.status_code}")

        data = response.json()
        content = data.get("response", "").strip()
        return _parse_json_result(content)

    except requests.exceptions.ConnectionError:
        raise Exception("Ollama server not running. Start with 'ollama serve'")
    except Exception as e:
        raise Exception(f"Ollama error: {str(e)}")


def _parse_json_result(content: str) -> dict:
    """Parse JSON result from LLM response."""
    # Handle markdown code blocks
    if content.startswith("```"):
        lines = content.split("\n")
        # Remove first and last lines (``` markers)
        content = "\n".join(lines[1:-1])

    # Try to find JSON object in response
    start_idx = content.find("{")
    end_idx = content.rfind("}")

    if start_idx != -1 and end_idx != -1:
        content = content[start_idx:end_idx + 1]

    result = json.loads(content)

    if not isinstance(result, dict) or "spoken" not in result:
        raise ValueError("Invalid response format - missing 'spoken' field")

    # Ensure changes is a list
    if "changes" not in result:
        result["changes"] = []

    return result


def normalize_to_spoken(text: str, provider: str = None) -> dict:
    """
    Normalize formal Turkish text to colloquial spoken form using LLM.

    Args:
        text: The formal written text to normalize
        provider: LLM provider to use (openai/ollama). Defaults to current.

    Returns:
        Dict with keys:
            - original: Original text
            - spoken: Normalized spoken form
            - changes: List of {from, to} changes made
    """
    if not text or not text.strip():
        return {"original": text, "spoken": text, "changes": []}

    active_provider = provider or _current_provider

    prompt = _build_normalization_prompt(text.strip())

    max_retries = 2
    for attempt in range(max_retries + 1):
        try:
            if active_provider == "ollama":
                result = _normalize_with_ollama(prompt)
            else:
                result = _normalize_with_openai(prompt)

            return {
                "original": text,
                "spoken": result["spoken"],
                "changes": result.get("changes", [])
            }

        except json.JSONDecodeError as e:
            print(f"⚠️ JSON parse error (attempt {attempt + 1}): {e}")
            if attempt == max_retries:
                print(f"❌ Failed to normalize, returning original text")
                return {"original": text, "spoken": text, "changes": []}

        except Exception as e:
            print(f"❌ Normalization error: {e}")
            return {"original": text, "spoken": text, "changes": []}

    return {"original": text, "spoken": text, "changes": []}


def batch_normalize(texts: List[str], provider: str = None) -> List[dict]:
    """
    Normalize multiple texts to colloquial spoken forms.

    Args:
        texts: List of formal texts to normalize
        provider: LLM provider to use

    Returns:
        List of normalization results
    """
    results = []
    for text in texts:
        result = normalize_to_spoken(text, provider)
        results.append(result)
        print(f"📝 '{text[:40]}...' → '{result['spoken'][:40]}...'")

    return results


def compare_forms(formal: str, spoken: str) -> List[dict]:
    """
    Compare formal and spoken forms word by word.

    Returns:
        List of {word_index, formal, spoken, changed} dicts
    """
    formal_words = formal.split()
    spoken_words = spoken.split()

    comparison = []
    max_len = max(len(formal_words), len(spoken_words))

    for i in range(max_len):
        f_word = formal_words[i] if i < len(formal_words) else ""
        s_word = spoken_words[i] if i < len(spoken_words) else ""

        comparison.append({
            "index": i,
            "formal": f_word,
            "spoken": s_word,
            "changed": f_word != s_word
        })

    return comparison


def get_settings() -> dict:
    """Get current colloquial normalizer settings."""
    return {
        "enabled": _colloquial_enabled,
        "provider": _current_provider
    }


def update_settings(enabled: bool = None, provider: str = None):
    """Update colloquial normalizer settings."""
    global _current_provider

    if enabled is not None:
        set_enabled(enabled)

    if provider is not None:
        _current_provider = provider
