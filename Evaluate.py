def translate(model, src_list, input_vocab, target_vocab, max_length=30):
    model.eval()
    translations = []

    with torch.no_grad():
        for src in src_list:
            src = torch.tensor(src, dtype=torch.long).unsqueeze(0).to(device)
            hidden = model.encoder(src)
            input_token = torch.tensor([target_vocab['<go>']], dtype=torch.long).to(device)
            translated_text = []

            for _ in range(max_length):
                output, hidden = model.decoder(input_token, hidden)
                top1 = output.argmax(1).item()
                
                if top1 == target_vocab['<eos>']:
                    break
                
                translated_text.append(top1)
                input_token = torch.tensor([top1], dtype=torch.long).to(device)
            
            translated_text = ''.join([list(target_vocab.keys())[list(target_vocab.values()).index(i)] for i in translated_text])
            translations.append(translated_text)
    
    return translations


def evaluate_accuracy(model, df, target_column, input_vocab, target_vocab, max_length=30):
    correct = 0
    total = 0

    src_list = df['src'].tolist()
    target_list = df[target_column].tolist()

    translations = translate(model, src_list, input_vocab, target_vocab, max_length)

    for translation, target in zip(translations, target_list):
        if translation == target:
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0
    return accuracy