# Importing the necessary libraries
import csv
import json
import openai
import pandas as pd

# Importing the dataset
df = pd.read_csv('./tweets.csv')
tweets = df['text']

# Defining prompts and labels

# Prompt 1 (Zero-Shot)
prompt1 = """Assess the sentiment and message conveyed in the tweet above about the corona virus pandemic handling. Choose a label from the options and provide a concise justification in JSON format, considering specific details and context."""

# Prompt 2 (Few-Shot)
prompt2 = f"""Here are some example tweets about the corona virus pandemic handling:

  * "So grateful for all the frontline workers who helped us get through this pandemic!" (Positive)
  * "Frustrated with the mixed messages from officials about the pandemic. Wish there was a clear plan." (Negative)
  * "Finally seeing a decrease in cases! Maybe we can finally move on from this." (Neutral)

Now, assess the sentiment and message conveyed in the following tweet about the corona virus pandemic handling:"""

# Prompt 3 (In-Context Learning)
prompt3 = f"""The COVID-19 pandemic has been ongoing for several years, with many countries now in various stages of recovery. 

Now, evaluate the overall sentiment and message in the following tweet regarding corona virus pandemic handling:

  * Consider whether the tweet reflects on past handling of the pandemic or the current situation.
  * Look for clues about the author's opinion on the effectiveness of the response."""

prompts = [prompt1, prompt2, prompt3]
labels = 'Positive, Extremely Positive, Negative, Extremely Negative'

# Creating OpenAI client
client = openai.OpenAI(api_key='your_api_key')

# Creating a request function
def request(tweet, labels, prompt):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": 'tweet: ' + tweet + ', labels: ' + labels + ', task:' + prompt +
                                         ', note give the output in the form {"label":"your_label","reason":"your justification"}'}
        ]
    )
    return completion.choices[0].message.content

# Generate requests and save label and reason in a file
for i, tweet in enumerate(tweets):
    data_row = []
    data_row.append(i)
    data_row.append(tweet)
    for prompt in prompts:
        data_row.append(prompt)
    comp = json.loads(request(tweet, labels, prompt))
    label, reason = comp['label'], comp['reason']
    data_row.append(label)
    data_row.append(reason)
    print(i, label, reason)
    with open('gpt-annotations.csv', 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(data_row)