# Läser filnamn och etiketter från filen
with open('labels.txt', 'r') as file:
    lines = file.readlines()

# Initialiserar listor för filnamn och etiketter
image_files = []
labels = []

# Itererar genom varje rad i filen
for line in lines:
    # Delar upp raden vid kommatecknet och avlägsnar vitrymmen
    image_file, label = line.strip().split(', ')
    # Lägger till filnamnet och etiketten (konverterad till int) i listorna
    image_files.append(image_file)
    labels.append((label)) #för int eller float ersätt med int(label) eller float(label)

# Visar extraherade filnamn och etiketter
print("Image Files:", image_files)
print("Labels:", labels)
