#scikit-learn kütüphanesi kullanarak iris veri setinden verileri yüklüyoruz, veri setini eğitim ve test kümelerine ayırıyoruz, Karar ağacı sınıflandırıcısını eğitiyoruz, test verileri üzerinde tahmin yapıyoruz ve tahminlerin doğruluğunu hesaplıyoruz.

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Veri setini yükle
iris = load_iris()
X = iris.data
y = iris.target

# Veri setini eğitim ve test kümelerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Karar ağacı sınıflandırıcısını eğit
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Test verileri üzerinde tahmin yap
y_pred = clf.predict(X_test)

# Tahminlerin doğruluğunu hesapla
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
