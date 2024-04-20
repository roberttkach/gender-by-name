from data_preprocessing import preprocess_data
from model_training import train_model
from gender_prediction import run_prediction


def main():
    csv_file = r'data\data.csv'
    names, genders_bin, longest_word = preprocess_data(csv_file)
    tokenizer = train_model(names, genders_bin, longest_word)
    run_prediction(tokenizer, longest_word)


if __name__ == '__main__':
    main()
