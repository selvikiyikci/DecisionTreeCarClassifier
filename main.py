import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Veri yükleme
train_data = pd.read_excel('trainDATA.xlsx')
test_data = pd.read_excel('testDATA.xlsx')

# Eğitim ve test verisinin ayrıştırılması
X_train = train_data.drop(columns=['Car Acceptibility'])
y_train = train_data['Car Acceptibility']

X_test = test_data  # Test verisi zaten hedef sütun içermiyor

# Karar ağacı modeli oluşturma ve eğitme
model = DecisionTreeClassifier(random_state=42, max_depth=5)  
model.fit(X_train, y_train)

# Karar ağacını görselleştirme
plt.figure(figsize=(30, 20)) 
ax = plt.gca() 
plot_tree(
    model,
    feature_names=X_train.columns,
    class_names=[str(c) for c in model.classes_],
    filled=True,
    fontsize=10,  # Yazı boyutunu artır
    impurity=False
)

ax.set_xlim(-1.5, len(X_train.columns) * 6)  
ax.set_ylim(-1.5, len(model.classes_) * 8)  

plt.title("Decision Tree Visualization", fontsize=18)  
plt.savefig("decision_tree_visualization.png", dpi=300)  
plt.show()

# Test verisi için tahminler
predictions = model.predict(X_test)

# Tahminleri "1" ve "2" olarak yeniden kodlama
class_mapping = {cls: str(i + 1) for i, cls in enumerate(model.classes_)}  # Sınıfları "1", "2" olarak eşleştir
predictions_mapped = [class_mapping[pred] for pred in predictions]

# Tahmin sonuçlarını kaydetme
test_data['Predicted Acceptibility'] = predictions_mapped
test_data.to_excel('test_results.xlsx', index=False)

print("Karar ağacı başarıyla inşa edildi, görselleştirildi ve tahmin sonuçları kaydedildi.")
