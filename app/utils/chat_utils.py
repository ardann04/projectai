GREETINGS = [
    # Bahasa Indonesia formal & santai
    "hai","haaii","haai", "haaai", "haii", "haiii", "haaaai", "haloo", "halooo", "haloooo", "halo", "hallo", "hallooo",
    "heii", "heiii", "hei", "heyy", "heyyy", "heyyyy", "heyyyyy", "hey", "hey!", "hi", "hii", "hiii", "hiiii", "hi!", 
    "selamat pagi", "selamat siang", "selamat sore", "selamat malam",
    "pagi", "siang", "sore", "malam",
    "apa kabar", "apa kabar?", "apa kabar!", "kabar baik?", "kabar baik",

    # Variasi "alo" & panjangnya
    "alo", "aloo", "alooo", "aloooo", "alooooo", "aloooooo", "alooooooo",
    "alo :)", "aloo :)", "alooo :)", "aloooo :)", "alooooo :)", "alo :D", "aloo :D", "alooo :D",
    
    # Bahasa Inggris umum
    "hello", "helloo", "heelloo", "heellooo", "hi there", "hi there!", "hey there", "hey there!", 
    "good morning", "good afternoon", "good evening", "morning", "afternoon", "evening",

    # Kasual / slang
    "yo", "yo!", "yo bro", "yo bro!", "sup", "sup?", "sup!", "hiya", "hiya!", "howdy", "heya", "heya!", 
    "wassup", "wassup?", "what's up", "what's up?", "whats up", "heyyo", "heyyo!", "hiiiii :)",
    
    # Variasi panjang & pengulangan huruf
    "haaaaai", "heeei", "heeey", "heeeey", "heeeeey", "hiiii", "hiiiii", "hiiiiii", "haloooooo", "hulloooo",
    
    # Emoticon & smiley
    "hi :)", "hi :D", "hello :)", "hey :)", "hii :)", "hii :D", "heyyy :)", "heyyy :D", "halo :)", "halo :D",
    
    # Lebih kasual & random manusiawi
    "hullo", "hulloo", "hullooo", "hay", "hayy", "hayyy", "hayyyy", "hiyaa", "hiyaaa", "heyhey", "heyhey!", "hiiiiiiiii",
    "yo yo", "yo yo!", "yo brooo", "yo broooo", "sup sup", "sup sup?", "wassuuup", "wassuuup?", "heeeyy :)", "heeeyy :D"
]

GREETING_RESPONSE = "Hai juga! 😊"

def handle_greetings(user_input):
    """
    Cek apakah input termasuk greetings sederhana atau panjang/pengulangan.
    """
    normalized = user_input.lower().strip()
    if normalized in GREETINGS:
        return GREETING_RESPONSE
    return None
