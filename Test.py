import openai
import pandas as pd

# Set up the API key
openai.api_key = "YOUR_API_KEY"

# Load the movie review dataset
reviews = pd.read_csv("dataset.csv")

# Split the data into training and validation sets
train_reviews = reviews[:int(0.8 * len(reviews))]
validation_reviews = reviews[int(0.8 * len(reviews)):]

# Train the model on the training data
for epoch in range(5):
    for i, review in train_reviews.iterrows():
        prompt = f"Summary of the movie review: {review['review']}"
        completion = openai.Completion.create(
            engine="davinci",
            prompt=prompt,
            max_tokens=1024,
            n=1,
            stop=None,
            temperature=0.5,
        )
        message = completion.choices[0].text
        # Save the generated summary to the training data
        train_reviews.at[i, "summary"] = message

# Evaluate the model on the validation data
for i, review in validation_reviews.iterrows():
    prompt = f"Summary of the movie review: {review['review']}"
    completion = openai.Completion.create(
        engine="davinci",
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.5,
    )
    message = completion.choices[0].text
    # Save the generated summary to the validation data
    validation_reviews.at[i, "summary"] = message

# Calculate the accuracy of the model on the validation data
accuracy = sum(validation_reviews["summary"] == validation_reviews["real_summary"]) / len(validation_reviews)
print("Accuracy:", accuracy)
