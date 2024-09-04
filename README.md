# Transileration_Eng_to_Tel
In this project, we’re working with the Dakshina dataset provided by Google, focusing on transliteration between scripts. I’ve chosen Telugu as the native language for this assignment.

The task involves creating a model that can convert romanized words (typed in Latin script) into their correct Telugu script equivalents. For example, when someone types “ghar” in English letters, the model should be able to produce the corresponding word “ఘర” in Telugu script.The dataset contains pairs of words like “ajanabee” and “అజనబీ,” where one word is in the Latin script, and the other is in Telugu. The goal is to train a model  \hat{f}(x)  that can learn this mapping from Latin to Telugu script at the character level.
This problem is similar to machine translation, but instead of translating sequences of words between languages, we’re focusing on translating sequences of characters from one script to another. The model should be able to generalize well to unseen words, making it useful for real-world applications like transliteration in chat applications.


This project uses the following datasets:

1. Dakshina Dataset
   Citation:  
   "Dakshina: A Multilingual Dataset for Transliteration." Google Research.  
   Available at: [https://github.com/google-research-datasets/dakshina](https://github.com/google-research-datasets/dakshina)
   Citation:  
   @inproceedings{roark-etal-2020-processing,
    title = "Processing {South} {Asian} Languages Written in the {Latin} Script:
    the {Dakshina} Dataset",
    author = "Roark, Brian and
      Wolf-Sonkin, Lawrence and
      Kirov, Christo and
      Mielke, Sabrina J. and
      Johny, Cibu and
      Demir{\c{s}}ahin, I{\c{s}}in and
      Hall, Keith",
    booktitle = "Proceedings of The 12th Language Resources and Evaluation Conference (LREC)",
    year = "2020",
    url = "https://www.aclweb.org/anthology/2020.lrec-1.294",
    pages = "2413--2423"
}
