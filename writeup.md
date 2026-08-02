
# Problem (look_at_cc)

(a)
The very first web page URL is http://020bld.cn.vauofnj.cn/html/991f998999.html. The page does not seem to be accessible anymore.
It's a Chinese page and the translated description is: "Can lily pollen stains be removed? - Industry Encyclopedia Method: Use a vacuum cleaner to remove most of the pollen from the area where it fell. etc.". ChatGPT helped me analyze that it might be a Chinese SEO landing page or scraped Q&A page.

(b)
The text is in Chinese again, but after translating we see the phrase "contact address" appearing multiple times in a row as well as a phone number appearing multiple times in a row for no apparent reason. The content of the article is also not very consistent and drifts from topic to topic. I think LLMs are generally trained on Chinese webpages as well, but perhaps the majority of the data is still in English. Despite the webpage containing lots of duplicates and topic drift, the model could still learn language structure and logic from this.

(c)
This example could be useful for training a classifier or finetuning an LLM to detect spam webpages. It may not be useful for general-purpose chat or instruction-following models.


# Problem (extract_text)

(a) See cs336_data/utils.py extract_text()

(b)
Both are pretty stimilar and extract the same core article content. However the text extraction function using resiliparse seems to keep more of the original structure of the article such as bulletpoints, spaces, and newlines, while the WET version is a lot more compact. The text extraction function seems slightly better.


# Problem (language_identification)

(a) See cs336_data/utils.py identify_language()

(b)
Language identification errors can introduce false positives and false negatives causing the model to learn vocab and syntax from other unintended languages, potentially lowering its performance on the main language it was supposed to be trained on. To mitigate these issues, one could use a pipeline of multiple language identification classifiers rather than relying on just a single classifier.


# Problem (mask_pii)

(a) See cs336_data/utils.py mask_emails()

(b) See cs336_data/utils.py mask_phone_numbers()

(c) See cs336_data/utils.py mask_ips()

(d)
Naive filtering can produce both false positives and false negatives. And replacing every match with the same placeholder can distort the data distribution and reduce the model's ability to understand or generate valid emails, phone numbers, or IP addresses. To mitigate these issues, one could replace sensitive values with realistic synthetic examples instead of a fixed placeholder token?


# Problem (harmful_content)

(a) See cs336_data/utils.py detect_nsfw()

(b) See cs336_data/utils.py detect_toxic_speech()

(c)
When these filters are applied to create the training set, the language model won't be exposed to nsfw or toxic speech much and so will be less capable of handling these topics appropriately. I think it would be best to still expose the model to some examples of nsfw and toxic speech content during pretraining so it is at least knowledgeable about these topics. Then during SFT, fine-tune the model to discuss nsfw topics appropriately and refuse requests to generate toxic speech.

