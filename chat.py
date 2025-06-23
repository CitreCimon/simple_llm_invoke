import time
from langchain_community.chat_models import ChatOllama
from langchain.schema import HumanMessage

# === Modell initialisieren ===
chat = ChatOllama(
    model="llama3.1:8b",  # llama3:instruct llama3.2:3b, llama3.2:1b, llama3.1:8b
    base_url="http://localhost:11434",  # falls Ollama in Docker läuft
    temperature=0,
    verbose=False
)

# === Frage eingeben ===
#frage = "Fasse den folgenden Artikel zusammen: München (standarddeutsch [ˈmʏnçn̩] ⓘ oder [ˈmʏnçən];[2] bairisch Mingaⓘ/? [ˈmɪŋ(ː)ɐ]) ist die Landeshauptstadt des Freistaates Bayern.[3] Sie ist mit 1,5 Millionen Einwohnern die bevölkerungsreichste Stadt Bayerns, die drittgrößte Gemeinde Deutschlands und mit 4.844 Einwohnern pro Quadratkilometer die am dichtesten bevölkerte Gemeinde Deutschlands. Verwaltungsrechtlich ist München eine kreisfreie Stadt. Sie bildet das Zentrum der Metropolregion München (rund 6,2 Millionen Einwohner)[4] und der Planungsregion München (2,93 Millionen Einwohner).[5] München wird zu den Weltstädten gezählt und gilt als ein Zentrum der Kultur, Politik, Wissenschaften und Medien.[6] München ist Sitz des Bayerischen Landtages, der Bayerischen Staatsregierung, Verwaltungssitz des die Stadt umgebenden Landkreises München mit dessen Landratsamt sowie des bayerischen Bezirks Oberbayern und des Regierungsbezirks Oberbayern. Hinzu kommen Bundesbehörden und -gerichte, mehrere Landesbehörden und internationale Behörden. Vor Ort bestehen zahlreiche Konzerne, Universitäten und Hochschulen, bedeutende Museen, Theater und die einzige Börse Bayerns. Durch eine große Anzahl sehenswerter Bauten samt geschützten Baudenkmälern und Ensembles, internationaler Sportveranstaltungen, Messen und Kongresse sowie das weltbekannte Oktoberfest ist die Stadt ein Anziehungspunkt für den internationalen Tourismus und eine der meistbesuchten Städte Europas. "
frage = "Was weißt du über München?" 
#frage = "Was ist die Hauptstadt von Deutschland?" 
frage_zeichen = len(frage)

# === Startzeit messen ===
start_time = time.time()

# === Antwort generieren ===
antwort = chat([HumanMessage(content=frage)]).content

# === Endzeit messen ===
end_time = time.time()

# === Metriken berechnen ===
antwort_zeichen = len(antwort)
dauer = end_time - start_time

# === Ausgabe ===
print("\n🧠 Antwort vom Modell:")
print(antwort)

print("\n📊 Statistik:")
print(f"Frage-Zeichenanzahl:   {frage_zeichen}")
print(f"Antwort-Zeichenanzahl: {antwort_zeichen}")
print(f"Antwortzeit:           {dauer:.2f} Sekunden")
