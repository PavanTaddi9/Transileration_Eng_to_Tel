# Transileration_Eng_to_Tel
In this project, we’re working with the Dakshina dataset provided by Google, focusing on transliteration between scripts. I’ve chosen Telugu as the native language for this assignment.

The task involves creating a model that can convert romanized words (typed in Latin script) into their correct Telugu script equivalents. For example, when someone types “ghar” in English letters, the model should be able to produce the corresponding word “ఘర” in Telugu script.

The dataset contains pairs of words like “ajanabee” and “అజనబీ,” where one word is in the Latin script, and the other is in Telugu. The goal is to train a model  \hat{f}(x)  that can learn this mapping from Latin to Telugu script at the character level.

This problem is similar to machine translation, but instead of translating sequences of words between languages, we’re focusing on translating sequences of characters from one script to another. The model should be able to generalize well to unseen words, making it useful for real-world applications like transliteration in chat applications.
