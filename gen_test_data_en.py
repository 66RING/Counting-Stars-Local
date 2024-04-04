import json
import random
import argparse

def get_sky(sky_dir):
    sky_file = open(sky_dir, "r", encoding="utf-8")
    sky = ""
    for i in sky_file.readlines():
        sky += "".join(i.strip())
    return sky

# stars = [3, 5, 9, 15, 19, 21, 26, 29, 35, 38, 42, 46, 49, 54, 58, 61, 66, 69, 74, 77, 81, 86, 89, 94, 97, 102, 107, 109, 113, 117, 122, 127, 130, 135, 139, 142, 145, 150, 153, 158, 162, 167, 171, 175, 178, 183, 185, 190, 194, 198, 201, 206, 211, 213, 217, 223, 227, 230, 235, 239, 243, 245, 249, 255]
stars = [i for i in range(0, 512, 3)]
# NOTE: shuffle
random.shuffle(stars)

system_prompt = ""
retrieval_question = "\n\n" + "\n\nOn this moonlit and misty night, the little penguin is looking up at the sky and concentrating fully on counting ★. Please help the little penguin collect the number of ★, for example: {\"little_penguin\": [x, x, x,...]}. The summation is not required, and the numbers in [x, x, x,...] represent the counted number of ★ by the little penguin. Only output the results in JSON format without any explanation."
scalar = 0.725
version = [[16, 32], [32, 32], [64, 32], [32, 16]]
version = [[16, 32], [32, 32], [64, 32], [32, 16]]
max_context_length = 128 * 1024

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, default="./harry_potter_all.txt")
    args = parser.parse_args()

    sky = get_sky(args.input)
    for m, n in version:
        line_count = 0
        interval = int(max_context_length/n)
        sky_size = [int(i*scalar) for i in range(interval, max_context_length+1, interval)]
        file_name = f"Counting_Stars_{m}_{n}.jsonl"
        test_data = open(file_name, "w", encoding="utf-8")
        print(file_name)
        for j in sky_size:
            indicator = 0
            sprinkle_stars_sky = sky[:j]
            for k in range(0, j, int(j / m)):
                star_number = stars[indicator]
                indicator += 1
                single_star = f"\nThe little penguin counted {star_number} ★ \n"
                sprinkle_stars_sky = (sprinkle_stars_sky[:k+int(j / m)] + single_star + sprinkle_stars_sky[k + int(j / m):])
                if indicator == m:
                    print(f"> Total stars count {indicator}")
                    break
            output_template = {"question": system_prompt + sprinkle_stars_sky + retrieval_question, "sky_size": j, "retrieval_question": retrieval_question,
                        "reference_counting_results": stars[:m], "parameters": {"temperature": 0.0, "frequency_penalty": 0.0, "presence_penalty": 0.0}}
            print(json.dumps(output_template, ensure_ascii=False), file=test_data)
            line_count += 1
            test_data.flush()
        test_data.close()
        print(f"Dataset size {line_count}")
