# Counting-Stars-Local

Fork from [Counting-Stars](https://github.com/nick7nlp/Counting-Stars).

## Usage

1. Prepare a long context like `./harry_potter_all.txt`
2. generate test data: `python gen_test_data_en.py`
    - You may need to change prompt to get it right
3. run evaluation: `python test -m <model_path>`
4. insert result label into json file like: `"answer": "{\"little_penguin\": [1, 2, 3, 4]}",`
5. generate visualized graph

## Terms

- M: How many stars
- N: How long the context should be
    * interval = MAX_LEN/N
    * sky_size = [1 x interval, 2 x interval, ..., N x interval]
        + each sky contain M stars

