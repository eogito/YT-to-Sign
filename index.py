import os
import subprocess
import stanza
from operator import itemgetter, attrgetter, methodcaller
import json

from youtube_transcript_api import YouTubeTranscriptApi

# Download models on first run
# stanfordnlp.download('en')   # This downloads the English models for the neural pipeline
# Sets up a neural pipeline in English
nlp = stanza.Pipeline(processors='tokenize,mwt,pos,lemma,depparse', treebank='en_ewt', use_gpu=False, pos_batch_size=3000) # Build the pipeline, specify part-of-speech processor's batch size

def parse(text):
  # Process text input
  doc = nlp(text) # Run the pipeline on text input

  print ("""
  ┌─┐┌─┐┬─┐┌─┐┌─┐┬─┐┌┬┐  ┌┬┐┬─┐┌─┐┌┐┌┌─┐┬  ┌─┐┌┬┐┬┌─┐┌┐┌
  ├─┘├┤ ├┬┘├┤ │ │├┬┘│││   │ ├┬┘├─┤│││└─┐│  ├─┤ │ ││ ││││
  ┴  └─┘┴└─└  └─┘┴└─┴ ┴   ┴ ┴└─┴ ┴┘└┘└─┘┴─┘┴ ┴ ┴ ┴└─┘┘└┘
  """)

  for sentence in doc.sentences:
  
    translation = translate(sentence)

    result = []
    for word in translation[0]:
      result.append((word['text'].lower(), word['lemma'].lower()))
    print("\nResult: ", result, "\n")

    print ("""
  ┌─┐┌─┐┬    ┬─┐┌─┐┌─┐┬─┐┌─┐┌─┐┌─┐┌┐┌┌┬┐┌─┐┌┬┐┬┌─┐┌┐┌
  ├─┤└─┐│    ├┬┘├┤ ├─┘├┬┘├┤ └─┐├┤ │││ │ ├─┤ │ ││ ││││
  ┴ ┴└─┘┴─┘  ┴└─└─┘┴  ┴└─└─┘└─┘└─┘┘└┘ ┴ ┴ ┴ ┴ ┴└─┘┘└┘
    """)
    display(translation)

  return doc

def wordToDictionary(word):
  dictionary = {
    'id': word.id,
    'head': word.head,
    'text': word.text.lower(),
    'lemma': word.lemma.lower(),
    'upos': word.upos,
    'xpos': word.xpos,
    'deprel': word.deprel,
    'feats': word.feats,
    'children': []
  }
  return dictionary



def getMeta(sentence):
  # sentence.print_dependencies()
  englishStruct = {}
  aslStruct = {
    'rootElements':[],
    'UPOS': {
      'ADJ':[], 'ADP':[], 'ADV':[], 'AUX':[], 'CCONJ':[], 'DET':[], 'INTJ':[], 'NOUN':[], 'NUM':[], 'PART':[], 'PRON':[], 'PROPN':[], 'PUNCT':[], 'SCONJ':[], 'SYM':[], 'VERB':[], 'X':[]
    }
  }
  reordered = []
  # aslStruct["rootElements"] = []

  # Make a list of all tokenized words. This step might be unnecessary.
  words = []
  for token in sentence.tokens:
    # print(token)
    for word in token.words:
      
      #print(word.index, word.governor, word.text, word.lemma, word.upos, word.deprel) # , word.feats)
      # # Insert as dict
      # words.append(wordToDictionary(word))
      # Insertion sort
      j = len(words)
      for i, w in enumerate(words):
        if word.head <= w['head']:
          continue
        else:
          j = i
          break
      # Convert to Python native structure when inserting.
      words.insert(j, wordToDictionary(word))
  # # Python sort for converted words
  # words.sort(key=attrgetter('governor', 'age')) # , reverse=True
  # words.sort(key=words.__getitem__) # , reverse=True
  reordered = words

  # Deprecated aslStruct code...
  # While there exist words that haven't been added to the tree.  
  # englishStruct['root'] = wordToDictionary(words[0])
  #     # Create list of words for each UPOS
  #     aslStruct['UPOS'][word.upos].append(word)
  # 
  # # Sort each UPOS list
  # # print(aslStruct['UPOS'])
  # for upos, uposList in aslStruct['UPOS'].items():
  #   # print(upos, uposList)
  #   uposList.sort(key=attrgetter('governor'))
  #   print(upos, uposList)

  # Identify Root Elements
  # for word in token.words:
    # if word.deprel == "root":
      # aslStruct["rootElements"].append(word)
      # Get related elements
      # Ident topics & comments

  # print("\n", aslStruct, "\n")
  return reordered

def getLemmaSequence(meta):
  tone = ""
  translation = []
  for word in meta:
    # Remove blacklisted words
    if (word['text'].lower(), word['lemma'].lower()) not in (('is', 'be'), ('the', 'the'), ('of', 'the'), ('is', 'are'), ('by', 'by'), (',', ','), (';', ';'), (':'), (':')):
      
      # Get Tone: get the sentence's tone from the punctuation
      if word['upos'] == 'PUNCT':
        if word['lemma'] == "?":
          tone = "?"
        elif word['lemma'] == "!":
          tone = "!"
        else:
          tone = ""
        continue
      
      # Remove symbols and the unknown
      elif word['upos'] == 'SYM' or word['upos'] == 'X':
        continue
      
      # Remove particles
      if word['upos'] == 'PART':
        continue

      # Convert proper nouns to finger spell
      elif word['upos'] == 'PROPN':
        fingerSpell = []
        for letter in word['text'].lower():
          print(letter)
          spell = {}
          spell['text'] = letter
          spell['lemma'] = letter
          # Add fingerspell as individual lemmas
          fingerSpell.append(spell)
        print(fingerSpell)
        translation.extend(fingerSpell)
        print(translation)

      # Numbers
      elif word['upos'] == 'NUM':
        fingerSpell = []
        for letter in word['text'].lower():
          spell = {}
          # Convert number to fingerspell
          pass
          # Add fingerspell as individual lemmas
          fingerSpell.append(spell)

      # Interjections usually use alternative or special set of signs
      elif word['upos'] == 'CCONJ':
        translation.append(word)
      
      # Interjections usually use alternative or special set of signs
      elif word['upos'] == 'SCONJ':
        if (word['text'].lower(), word['lemma'].lower() not in (('that', 'that'))):
          translation.append(word)
      
      # Interjections usually use alternative or special set of signs
      elif word['upos'] == 'INTJ':
        translation.append(word)

      # Adpositions could modify nouns
      elif word['upos']=='ADP':
        # translation.append(word)
        pass

      # Determinants modify noun intensity
      elif word['upos']=='DET':
        pass

      # Adjectives modify nouns and verbs
      elif word['upos']=='ADJ':
        translation.append(word)
        # pass

      # Pronouns
      elif word['upos'] == 'PRON' and word['deprel'] not in ('nsubj'):
        translation.append(word)

      # Nouns
      elif word['upos'] == 'NOUN':
        translation.append(word)

      # Adverbs modify verbs, leave for wh questions
      elif word['upos']=='ADV':
        translation.append(word)
      
      elif word['upos']=='AUX':
        pass

      # Verbs
      elif word['upos']=='VERB':
        translation.append(word)

  # translation = tree
  return (translation, tone)

def translate(parse):
  meta = getMeta(parse)
  translation = getLemmaSequence(meta)
  return translation



def display(translation):
  print(translation)
  folder = os.getcwd()
  filePrefix = folder + "/asl/"
  # Alter ASL lemmas to match sign's file names.  
  # In production, these paths would be stoxred at the dictionary's database.
  files = [ filePrefix + get_id(word['text'].lower())  + ".mp4" for word in translation[0] ]
  # Run video sequence using the MLT Multimedia Framework
  
  print("Running command: ", ["mpv"] + files)
  process = subprocess.Popen(["mpv"] + files, stdout=subprocess.PIPE)
  result = process.communicate()


f = open ('WLASL_v0.3.json', "r")
data = json.loads(f.read())

folder = os.getcwd()
filePrefix = folder + "/asl/"
def get_id(id):
  vid = next(
      (item for item in data if item["gloss"] == id), None
  )
  if vid:
      print(vid["instances"]  )
      for x in vid["instances"] :
        print(filePrefix+x["video_id"]+'.mp4')
        if (os.path.isfile(filePrefix+x["video_id"]+'.mp4')) :
          return x["video_id"]
  return "hello"
  
tests = [           
]

def get_captions(video_id, language='en'):
    try:
        # Fetch the transcript for the given video ID and language
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        
        # Find the transcript in the desired language
        transcript = transcript_list.find_transcript([language])

        # Fetch the transcript text
        captions = transcript.fetch()

        # Save the captions to a file
        for caption in captions:
            start = caption['start']
            duration = caption['duration']
            text = caption['text']
            tests.append(f"{text}")
            print("{text}")

    except Exception as e:
        print(f"An error occurred: {e}")

video_id = "VAMn67MmT24"  # Extract the video ID from the URL
get_captions(video_id, language='en')
print(tests)

for text in tests:

    print("Text to process: ", text, "\n")

    parse(text)