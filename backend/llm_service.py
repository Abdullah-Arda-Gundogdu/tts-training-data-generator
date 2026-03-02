"""
LLM Service - Multi-Provider Support for Sentence Generation

Supports OpenAI GPT and Ollama for generating synthetic Turkish sentences
containing mispronounced words for TTS training data.
"""

import os
import json
import requests
from typing import List, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Provider configuration
_current_provider = os.getenv("LLM_PROVIDER", "openai")
_ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
_ollama_model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

# OpenAI client (lazy initialized)
_openai_client = None


def get_current_config():
    """Get current LLM configuration."""
    return {
        "provider": _current_provider,
        "ollama_base_url": _ollama_base_url,
        "ollama_model": _ollama_model,
        "openai_available": bool(os.getenv("OPENAI_API_KEY")),
        "ollama_available": _check_ollama_available()
    }


def set_provider(provider: str, model: str = None):
    """Set the active LLM provider."""
    global _current_provider, _ollama_model
    
    if provider not in ["openai", "ollama"]:
        raise ValueError(f"Invalid provider: {provider}. Use 'openai' or 'ollama'")
    
    _current_provider = provider
    if model and provider == "ollama":
        _ollama_model = model
    
    print(f"✅ LLM provider set to: {provider}" + (f" (model: {model})" if model else ""))
    return get_current_config()


def _check_ollama_available():
    """Check if Ollama server is running."""
    try:
        response = requests.get(f"{_ollama_base_url}/api/tags", timeout=2)
        return response.status_code == 200
    except:
        return False


def get_ollama_models():
    """Get list of available Ollama models."""
    try:
        response = requests.get(f"{_ollama_base_url}/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return [model["name"] for model in data.get("models", [])]
        return []
    except Exception as e:
        print(f"❌ Failed to get Ollama models: {e}")
        return []


def _init_openai_client():
    """Initialize OpenAI client."""
    global _openai_client
    from openai import OpenAI
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    _openai_client = OpenAI(api_key=api_key)
    print("✅ OpenAI client initialized")
    return _openai_client


def _get_openai_client():
    """Get or initialize OpenAI client."""
    global _openai_client
    if _openai_client is None:
        _init_openai_client()
    return _openai_client


def _build_prompt(word: str, count: int, context: str = None, existing_sentences: List[str] = None, language: str = "Turkish", system_prompt: str = None):
    """Build the prompt for sentence generation."""
    context_instruction = f"Cümleler {context} alanıyla ilgili olsun." if context else ""
    
    existing_instruction = ""
    if existing_sentences:
        existing_list = "\n".join([f"- {s}" for s in existing_sentences[-20:]])
        existing_instruction = f"""
ÖNEMLİ: Aşağıdaki mevcut cümleleri TEKRARLAMA veya benzerini üretme:
{existing_list}
"""

    user_instruction = ""
    if system_prompt:
        user_instruction = f"\nKULLANICI YÖNERGESİ: {system_prompt}\n"
    
    return f"""Sen bir Türkçe dil uzmanısın. Bir TTS (Metin-Konuşma) modeli eğitmek için, teknik veya spesifik kelimeleri içeren senkronize Türkçe cümleler üretiyorsun.

Tam olarak {count} adet anlamlı, açıklayıcı ve doğal Türkçe cümle üret. Her cümle "{word}" ifadesini içermelidir.

KRİTİK KURALLAR:
- "{word}" ifadesini BİREBİR AYNI şekilde kullan — büyük/küçük harf, tire, numara ve yazılışı koru. Değiştirme.
- İfade tek kelime veya birden fazla kelime olabilir, bütün olarak kullan.
- Her cümle 25-30 kelime uzunluğunda olmalıdır.  kısa veya kelime eksikliği olan cümleler ÜRETME.
- {word} ifadesi eğer bir kısaltma, havacılık, teknik veya mühendislik terimiyse, cümleyi sanki bir kullanım kılavuzundan, eğitim kitabından veya teknik bir prosedürden alınmış mantıklı bir bağlamda kur.
- Sadece "Şu {word} kelimesi şöyledir" gibi basit, anlamsız cümleler kurma. Kelimenin gerçek hayattaki veya teknik alandaki işlevini yansıtan cümleler kur.
- "{word}" ifadesini cümlenin SADECE SONUNDA kullan.
- Cümleler sesli okunmaya uygun ve akıcı olmalıdır.
- HER KELİME EN AZ 20 KELİME OLMALI

{context_instruction}
{existing_instruction}
{user_instruction}
ZORUNLU: Sadece geçerli bir JSON dizisi döndür. Açıklama, pre-text, markdown (```json gibi) veya başka bir şey YAZMA. Sadece listeyi dön.

Format:
["Birinci cümle burada.", "İkinci cümle burada.", "Üçüncü cümle burada."]"""


def _generate_with_openai(prompt: str, temperature: float = 0.8, max_tokens: int = 2000, is_full_prompt: bool = False) -> List[str]:
    """Generate sentences using OpenAI."""
    client = _get_openai_client()
    
    if is_full_prompt:
        # Full prompt provided by user — send as system message for max compliance
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "Yukarıdaki yönergeyi uygula ve cümleleri üret."}
        ]
    else:
        messages = [
            {
                "role": "system",
                "content": "Sen bir Türkçe TTS eğitim verisi uzmanısın. Doğal, günlük konuşmaya yakın Türkçe cümleler üretirsin. Sadece geçerli JSON dizisi döndür, başka hiçbir şey yazma."
            },
            {"role": "user", "content": prompt}
        ]
    
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )
    
    content = response.choices[0].message.content.strip()
    return _parse_json_response(content)


def _generate_with_ollama(prompt: str, temperature: float = 0.8, is_full_prompt: bool = False) -> List[str]:
    """Generate sentences using Ollama."""
    try:
        # For Ollama, full_prompt is sent as system message via /api/chat
        if is_full_prompt:
            response = requests.post(
                f"{_ollama_base_url}/api/chat",
                json={
                    "model": _ollama_model,
                    "messages": [
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": "Yukarıdaki yönergeyi uygula ve cümleleri üret."}
                    ],
                    "stream": False,
                    "options": {"temperature": temperature}
                },
                timeout=120
            )
        else:
            response = requests.post(
                f"{_ollama_base_url}/api/generate",
                json={
                    "model": _ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": temperature}
                },
                timeout=120
            )
        
        if response.status_code != 200:
            raise Exception(f"Ollama API error: {response.status_code}")
        
        data = response.json()
        content = data.get("message", {}).get("content", "").strip() if is_full_prompt else data.get("response", "").strip()
        return _parse_json_response(content)
        
    except requests.exceptions.ConnectionError:
        raise Exception("Ollama server not running. Start with 'ollama serve'")
    except Exception as e:
        raise Exception(f"Ollama error: {str(e)}")


def _parse_json_response(content: str) -> List[str]:
    """Parse JSON array from LLM response."""
    # Handle markdown code blocks
    if content.startswith("```"):
        lines = content.split("\n")
        content = "\n".join(lines[1:-1])
    
    # Try to find JSON array in response
    start_idx = content.find("[")
    end_idx = content.rfind("]")
    
    if start_idx != -1 and end_idx != -1:
        content = content[start_idx:end_idx + 1]
    
    sentences = json.loads(content)
    
    if not isinstance(sentences, list):
        raise ValueError("Response is not a list")
    
    return sentences


def generate_sentences(
    word: str,
    count: int = 5,
    context: Optional[str] = None,
    language: str = "Turkish",
    provider: str = None,
    system_prompt: str = None,
    full_prompt: str = None
) -> List[str]:
    """
    Generate natural sentences containing the specified word.
    
    Args:
        word: The mispronounced word to include in sentences
        count: Number of sentences to generate
        context: Optional context/domain (e.g., "aviation", "technical")
        language: Language for sentences (default: Turkish)
        provider: Override the default provider (openai/ollama)
        system_prompt: Optional extra instruction appended to default prompt
        full_prompt: Optional full prompt override (replaces default prompt entirely)
    
    Returns:
        List of generated sentences containing the word
    """
    active_provider = provider or _current_provider
    count = max(1, count)
    
    all_sentences = []
    batch_size = 10
    max_retries_per_batch = 3
    
    while len(all_sentences) < count:
        remaining = count - len(all_sentences)
        current_batch_size = min(batch_size, remaining)
        
        use_full = bool(full_prompt)
        if full_prompt:
            # Use the full prompt directly, replacing placeholders
            prompt = full_prompt.replace("{word}", word).replace("{count}", str(current_batch_size))
            # Append existing sentences dedup instruction if needed
            if all_sentences:
                existing_list = "\n".join([f"- {s}" for s in all_sentences[-20:]])
                prompt += f"\n\nÖNEMLİ: Aşağıdaki mevcut cümleleri TEKRARLAMA veya benzerini üretme:\n{existing_list}"
            print(f"📝 Using FULL user prompt ({len(prompt)} chars)")
        else:
            prompt = _build_prompt(
                word=word,
                count=current_batch_size,
                context=context,
                existing_sentences=all_sentences,
                language=language,
                system_prompt=system_prompt
            )
            print(f"📝 Using default _build_prompt ({len(prompt)} chars)")
        
        retries = 0
        batch_sentences = []
        
        while retries < max_retries_per_batch and len(batch_sentences) < current_batch_size:
            try:
                if active_provider == "ollama":
                    sentences = _generate_with_ollama(prompt, is_full_prompt=use_full)
                else:
                    sentences = _generate_with_openai(prompt, is_full_prompt=use_full)
                
                # Validate sentences contain the word
                word_lower = word.lower()
                valid_sentences = [s for s in sentences if word_lower in s.lower()]
                
                # Filter duplicates
                for s in valid_sentences:
                    if s not in all_sentences and s not in batch_sentences:
                        batch_sentences.append(s)
                        if len(batch_sentences) >= current_batch_size:
                            break
                
                if len(batch_sentences) < current_batch_size:
                    retries += 1
                    print(f"⚠️ Got {len(batch_sentences)}/{current_batch_size} valid sentences, retrying... ({retries}/{max_retries_per_batch})")
                else:
                    break
                    
            except json.JSONDecodeError as e:
                print(f"❌ Failed to parse response as JSON: {e}")
                retries += 1
            except Exception as e:
                print(f"❌ Error generating sentences: {e}")
                retries += 1
        
        all_sentences.extend(batch_sentences)
        print(f"✅ Generated {len(batch_sentences)} sentences for '{word}' using {active_provider} (total: {len(all_sentences)}/{count})")
        
        if len(batch_sentences) == 0:
            print(f"⚠️ Could not generate more sentences, stopping at {len(all_sentences)}")
            break
    
    return all_sentences[:count]


def regenerate_single_sentence(
    word: str,
    existing_sentences: List[str],
    context: Optional[str] = None,
    language: str = "Turkish",
    provider: str = None
) -> str:
    """Generate a single new sentence different from existing ones."""
    active_provider = provider or _current_provider
    
    existing_text = "\n".join([f"- {s}" for s in existing_sentences])
    
    prompt = f"""Aşağıdaki kelimeyi/ifadeyi içeren TEK BİR yeni Türkçe cümle üret: "{word}"

KRİTİK: "{word}" ifadesini BİREBİR AYNI şekilde kullan.

Cümle şu özelliklerde olsun:
- 15-25 kelime uzunluğunda (çok kısa olmasın, en az 8 saniye sürecek şekilde)
- Doğal ve günlük konuşmaya yakın
- Aşağıdaki mevcut cümlelerden FARKLI:
{existing_text}

Sadece cümle metnini döndür, başka hiçbir şey yazma."""

    try:
        if active_provider == "ollama":
            response = requests.post(
                f"{_ollama_base_url}/api/generate",
                json={
                    "model": _ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.9}
                },
                timeout=60
            )
            
            if response.status_code != 200:
                raise Exception(f"Ollama error: {response.status_code}")
            
            sentence = response.json().get("response", "").strip()
        else:
            client = _get_openai_client()
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You generate single sentences. Respond with just the sentence."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.9,
                max_tokens=150
            )
            sentence = response.choices[0].message.content.strip()
        
        # Remove quotes if present
        sentence = sentence.strip('"\'')
        
        print(f"✅ Regenerated sentence for '{word}': {sentence[:50]}...")
        return sentence
        
    except Exception as e:
        print(f"❌ Error regenerating sentence: {e}")
        raise


