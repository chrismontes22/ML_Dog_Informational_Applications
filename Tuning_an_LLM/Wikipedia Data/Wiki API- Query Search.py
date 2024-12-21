"""This code grabs the majority of the dog texts from wikikpedia
to improve results I added the word dog after most breeds, so that it would be more
accurate when it searched the breed (ex. 'Boxer' vs 'Boxer dog')
This code prints out a list of the classes that didnt work in order to make adjustments to the query"""


import wikipedia


# Insert dog breeds here as a list and run code. It will seearch wikipedia as a query and download
dog_breeds = ['Afghan Hound', 'African Wild Dog', 'Airedale Dog', 'American Hairless Dog', 'American Spaniel Dog', 'Basenji Dog', 'Basset Dog', 'Beagle Dog', 'Bearded Collie Dog', 'Bermaise Dog', 'Bichon Frise Dog', 'Blenheim Dog', 'Bloodhound Dog', 'Bluetick Dog', 'Border Collie Dog', 'Borzoi Dog', 'Boston Terrier Dog', 'Boxer Dog', 'Bull Mastiff Dog', 'Bull Terrier Dog', 'Bulldog Dog', 'Cairn Dog', 'Chihuahua Dog', 'Chinese Crested Dog', 'Chow Dog', 'Clumber Dog', 'Cockapoo Dog', 'Cocker Dog', 'Collie Dog', 'Corgi Dog', 'Coyote Dog', 'Dalmation Dog', 'Dingo Dog', 'Doberman Dog', 'Elk Hound Dog', 'French Bulldog', 'German Sheperd Dog', 'Golden Retriever Dog', 'Great Dane Dog', 'Great Perenees Dog', 'Greyhound Dog', 'Groenendael Dog', 'Irish Spaniel Dog', 'Irish Wolfhound Dog', 'Japanese Spaniel Dog', 'Komondor Dog', 'Labradoodle Dog', 'Labrador Dog', 'Lhasa Dog', 'Malinois Dog', 'Maltese Dog', 'Mex Hairless Dog', 'Newfoundland Dog', 'Pekinese Dog', 'Pit Bull Dog', 'Pomeranian Dog', 'Poodle Dog', 'Pug Dog', 'Rhodesian Dog', 'Rottweiler Dog', 'Saint Bernard Dog', 'Samoyed Dog', 'Schnauzer Dog', 'Scotch Terrier Dog', 'Shar Pei Dog', 'Shiba Inu Dog', 'Shih Tzu Dog', 'Siberian Husky Dog', 'Vizsla Dog', 'Yorkie Dog']


# List to keep track of breeds that couldn't be fetched
skipped_breeds = []

for dogbreed in dog_breeds:
    try:
        # Get the content of the page
        content = wikipedia.page(dogbreed).content

        # Open a file in write mode
        with open(f'{dogbreed}.txt', 'w', encoding='utf-8') as f:
            # Write the content to the file
            f.write(content)
    except wikipedia.exceptions.PageError:
        print(f"Could not fetch the page for {dogbreed}. Skipping...")
        skipped_breeds.append(dogbreed)

# Print the breeds that were skipped
print("\nThe following breeds were skipped:")
for breed in skipped_breeds:
    print(breed)
