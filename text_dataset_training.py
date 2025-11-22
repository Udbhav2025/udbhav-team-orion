from datasets import load_dataset

dataset = load_dataset("dmitva/human_ai_generated_text")

print(dataset)
train_data = dataset['train']
print(train_data.column_names)
#for i in range(3):
    #print(f"Text {i+1}: {train_data[i]['text']}")
    #print(f"Label {i+1}: {train_data[i]['label']}\n")

#import pandas as pd
#df = pd.DataFrame(train_data)

#print(df.head())