% factory_reports_classification.m
% Example script for text classification of factory reports

%% Sample data
texts = [
    "Motorun fazla ısındığı gözlemlendi ve durduruldu.";
    "Konveyör bantta beklenmeyen bir titreşim tespit edildi.";
    "Robot kolunun kalibrasyonu bozuldu, yeniden ayarlanacak.";
    "Sensör arızası nedeniyle üretim hattı geçici olarak durdu.";
    "Yıllık bakım tamamlandı, ekipman sorunsuz çalışıyor.";
    "Periyodik yağlama işlemleri başarıyla yapıldı.";
    "Filtreler değiştirildi ve sistem temizlendi.";
    "Planlı bakım nedeniyle üretim bir saat durduruldu.";
    "Günlük üretim miktarı hedefin üzerinde gerçekleşti.";
    "Kalite kontrol sonuçları olumlu, hata oranı düşük.";
    "Yeni vardiya sistemi verimliliği artırdı.";
    "Enerji tüketimi geçen aya göre azaldı." ];

labels = [
    "Ariza";
    "Ariza";
    "Ariza";
    "Ariza";
    "Bakim";
    "Bakim";
    "Bakim";
    "Bakim";
    "Rapor";
    "Rapor";
    "Rapor";
    "Rapor" ];

% Create table
reports = table(texts, labels, 'VariableNames', {'Text', 'Category'});

%% Split data
rng(1); % for reproducibility
cv = cvpartition(height(reports), 'HoldOut', 0.3);
trainData = reports(training(cv), :);
testData  = reports(test(cv), :);

%% Preprocess text
documentsTrain = tokenizedDocument(trainData.Text);
documentsTest  = tokenizedDocument(testData.Text);

bag = bagOfWords(documentsTrain);
bag = tfidf(bag);

XTrain = bag.Counts;
YTrain = trainData.Category;

bagTest = bagOfWords(documentsTest, bag.Vocabulary);
XTest = tfidf(bagTest).Counts;
YTest = testData.Category;

%% Train classifier
mdl = fitcecoc(XTrain, YTrain);

%% Evaluate
predLabels = predict(mdl, XTest);
accuracy = mean(predLabels == YTest);

disp("Test Accuracy: " + accuracy)

