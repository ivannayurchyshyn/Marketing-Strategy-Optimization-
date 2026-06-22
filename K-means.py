import os
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


file_path = r"C:\Users\Ivann\OneDrive - lnu.edu.ua\4курс\Курсова\DATA_fixed_clustering_model.xlsx"

if not os.path.exists(file_path):
    raise FileNotFoundError(f"Файл не знайдено: {file_path}")

print("=" * 70)
print("Файл знайдено:")
print(file_path)
print("=" * 70)


output_dir = r"C:\Users\Ivann\OneDrive - lnu.edu.ua\4курс\Курсова\КОД"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "K_means_results.xlsx")

print("\nФайл результатів буде збережено у папку:")
print(output_dir)
sheet_name = "Для кластеризації"

xls = pd.ExcelFile(file_path)

print("\nЛисти у файлі:")
for sheet in xls.sheet_names:
    print("-", sheet)

if sheet_name not in xls.sheet_names:
    raise ValueError(f"У файлі немає листа '{sheet_name}'.")

df = pd.read_excel(file_path, sheet_name=sheet_name)

print("\nВикористано лист:", sheet_name)
print("\nПерші рядки таблиці:")
print(df.head())
print("\nНазви колонок:")
print(df.columns.tolist())
print("\nРозмір таблиці:", df.shape)

required_columns = [
    "customer_id",
    "frequency_orders",
    "quantity_total",
    "revenue",
    "avg_order_value",
    "recency_days",
    "unique_products",
]

missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    raise ValueError(
        "У файлі не знайдено потрібні колонки: "
        + ", ".join(missing_columns)
    )

customers = df[required_columns].copy()

numeric_columns = [
    "frequency_orders",
    "quantity_total",
    "revenue",
    "avg_order_value",
    "recency_days",
    "unique_products",
]

for col in numeric_columns:
    customers[col] = pd.to_numeric(customers[col], errors="coerce")

customers = customers.dropna(subset=numeric_columns)

customers = customers[
    (customers["frequency_orders"] > 0)
    & (customers["quantity_total"] > 0)
    & (customers["revenue"] > 0)
    & (customers["avg_order_value"] > 0)
    & (customers["recency_days"] >= 0)
    & (customers["unique_products"] > 0)
].copy()

customers["quantity_per_order"] = (
    customers["quantity_total"] / customers["frequency_orders"]
)

customers["revenue_per_product"] = (
    customers["revenue"] / customers["unique_products"]
)

print("\nКількість клієнтів після очищення:", len(customers))

if len(customers) < 10:
    raise ValueError("Після очищення залишилося занадто мало клієнтів.")


features = [
    "recency_days",
    "frequency_orders",
    "revenue",
]

print("\nОзнаки для кластеризації:")
for feature in features:
    print("-", feature)

X = customers[features].copy()

X_log = np.log1p(X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_log)

print("\nЛогарифмування та стандартизацію виконано.")


min_k = 3
max_k = min(10, len(customers) - 1)

k_values = list(range(min_k, max_k + 1))

inertias = []
silhouette_scores = []

for k in k_values:
    model = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = model.fit_predict(X_scaled)

    inertias.append(model.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, labels))

k_results = pd.DataFrame(
    {
        "k": k_values,
        "inertia": inertias,
        "silhouette": silhouette_scores,
    }
)

print("\nОцінка кількості кластерів:")
print(k_results)


x = np.array(k_values)
y = np.array(inertias)

x1, y1 = x[0], y[0]
x2, y2 = x[-1], y[-1]

distances = []

for xi, yi in zip(x, y):
    numerator = abs(
        (y2 - y1) * xi
        - (x2 - x1) * yi
        + x2 * y1
        - y2 * x1
    )

    denominator = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
    distance = numerator / denominator
    distances.append(distance)

optimal_k = int(x[np.argmax(distances)])

chosen_silhouette = k_results.loc[
    k_results["k"] == optimal_k,
    "silhouette",
].values[0]

print("\nАвтоматично визначена кількість кластерів за методом ліктя:")
print(optimal_k)

print("\nSilhouette score для обраної кількості кластерів:")
print(round(chosen_silhouette, 4))


plt.figure(figsize=(8, 5))
plt.plot(k_results["k"], k_results["inertia"], marker="o")
plt.axvline(optimal_k, linestyle="--", label=f"Обране k = {optimal_k}")
plt.xlabel("Кількість кластерів")
plt.ylabel("Inertia")
plt.title("Метод ліктя для визначення кількості кластерів")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(k_results["k"], k_results["silhouette"], marker="o")
plt.axvline(optimal_k, linestyle="--", label=f"Обране k = {optimal_k}")
plt.xlabel("Кількість кластерів")
plt.ylabel("Silhouette score")
plt.title("Силуетний аналіз для перевірки якості кластеризації")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=20)

customers["cluster"] = kmeans.fit_predict(X_scaled) + 1

print("\nКластеризацію K-Means виконано.")

print("\nКількість клієнтів у кожному кластері:")
print(customers["cluster"].value_counts().sort_index())


profile_features = [
    "recency_days",
    "frequency_orders",
    "revenue",
    "avg_order_value",
    "unique_products",
    "quantity_total",
    "quantity_per_order",
    "revenue_per_product",
]

cluster_profile = customers.groupby("cluster")[profile_features].mean().round(2)

cluster_profile["customers_count"] = (
    customers.groupby("cluster")["customer_id"].count()
)

cluster_profile["customers_share_percent"] = (
    cluster_profile["customers_count"] / len(customers) * 100
).round(2)

print("\nПрофіль кластерів:")
print(cluster_profile)


# ============================================================
# 13. РОЗРАХУНОК ПРИВАБЛИВОСТІ СЕГМЕНТІВ
# ============================================================

profile_for_score = cluster_profile.copy()

score = pd.Series(0, index=profile_for_score.index, dtype=float)

# recency_days: менше значення краще.
score += profile_for_score["recency_days"].rank(ascending=False)

# інші показники: більше значення краще.
score += profile_for_score["frequency_orders"].rank(ascending=True)
score += profile_for_score["revenue"].rank(ascending=True)
score += profile_for_score["avg_order_value"].rank(ascending=True)
score += profile_for_score["unique_products"].rank(ascending=True)

profile_for_score["segment_attractiveness"] = score

cluster_profile["segment_attractiveness"] = (
    profile_for_score["segment_attractiveness"]
)


sorted_clusters = cluster_profile.sort_values(
    "segment_attractiveness",
    ascending=False,
).index.tolist()

segment_labels = [
    "Високоцінні клієнти",
    "Активні середні клієнти",
    "Нові або перспективні клієнти",
    "Клієнти з високим середнім чеком",
    "Нерегулярні клієнти",
    "Пасивні / низькоцінні клієнти",
]

segment_names = {}

for i, cl in enumerate(sorted_clusters):
    if i < len(segment_labels):
        segment_names[cl] = segment_labels[i]
    else:
        segment_names[cl] = "Додатковий сегмент"

customers["segment_name"] = customers["cluster"].map(segment_names)
cluster_profile["segment_name"] = cluster_profile.index.map(segment_names)


# ============================================================
# 15. МАРКЕТИНГОВІ РЕКОМЕНДАЦІЇ
# ============================================================

recommendations = {
    "Високоцінні клієнти": (
        "Найбільш цінний сегмент. Доцільно застосовувати програми лояльності, "
        "персональні пропозиції, преміальні товарні добірки та індивідуальні знижки."
    ),
    "Активні середні клієнти": (
        "Клієнти з регулярною купівельною поведінкою. Доцільно застосовувати "
        "email-маркетинг, крос-продажі, персоналізовані рекомендації та акційні пропозиції."
    ),
    "Нові або перспективні клієнти": (
        "Клієнти з потенціалом подальшого розвитку. Доцільно використовувати "
        "welcome-пропозиції, ремаркетинг і стимулювання повторних покупок."
    ),
    "Клієнти з високим середнім чеком": (
        "Клієнти, які можуть купувати рідше, але мають вищу вартість замовлення. "
        "Доцільно пропонувати набори товарів, преміальні пропозиції та персональні рекомендації."
    ),
    "Нерегулярні клієнти": (
        "Клієнти з нестабільною активністю. Доцільно використовувати нагадування, "
        "сезонні пропозиції, обмежені акції та недорогі канали комунікації."
    ),
    "Пасивні / низькоцінні клієнти": (
        "Найменш активний сегмент. Доцільно застосовувати реактиваційні кампанії "
        "з обмеженим бюджетом або не надавати цьому сегменту високого пріоритету."
    ),
    "Додатковий сегмент": (
        "Потребує додаткового аналізу та тестування окремих маркетингових дій."
    ),
}

cluster_profile["marketing_recommendation"] = (
    cluster_profile["segment_name"].map(recommendations)
)


# ============================================================
# 16. ПРІОРИТЕТ БЮДЖЕТУ
# ============================================================

def define_budget_priority(score_value):
    if score_value >= cluster_profile["segment_attractiveness"].quantile(0.70):
        return "Високий пріоритет"
    if score_value >= cluster_profile["segment_attractiveness"].quantile(0.35):
        return "Середній пріоритет"
    return "Низький пріоритет"


cluster_profile["budget_priority"] = (
    cluster_profile["segment_attractiveness"].apply(define_budget_priority)
)


# ============================================================
# 17. ТАБЛИЦЯ ПРОФІЛЮ КЛАСТЕРІВ
# ============================================================

cluster_profile_output = cluster_profile.reset_index()

cluster_profile_output = cluster_profile_output[
    [
        "cluster",
        "segment_name",
        "customers_count",
        "customers_share_percent",
        "recency_days",
        "frequency_orders",
        "revenue",
        "avg_order_value",
        "unique_products",
        "quantity_total",
        "quantity_per_order",
        "revenue_per_product",
        "segment_attractiveness",
        "budget_priority",
        "marketing_recommendation",
    ]
]

print("\nІнтерпретація кластерів:")
print(
    cluster_profile_output[
        [
            "cluster",
            "segment_name",
            "customers_count",
            "customers_share_percent",
            "recency_days",
            "frequency_orders",
            "revenue",
            "segment_attractiveness",
            "budget_priority",
        ]
    ]
)


# ============================================================
# 18. PCA ДЛЯ ВІЗУАЛІЗАЦІЇ
# ============================================================

pca = PCA(n_components=2)
pca_components = pca.fit_transform(X_scaled)

customers["PC1"] = pca_components[:, 0]
customers["PC2"] = pca_components[:, 1]

explained_variance = pca.explained_variance_ratio_

pc1_var = explained_variance[0] * 100
pc2_var = explained_variance[1] * 100
total_var = pc1_var + pc2_var

print("\nPCA:")
print(f"PC1: {pc1_var:.2f}%")
print(f"PC2: {pc2_var:.2f}%")
print(f"Разом: {total_var:.2f}%")


# ============================================================
# 19. PCA-ГРАФІК 2D
# ============================================================

plt.figure(figsize=(9, 6))
scatter = plt.scatter(
    customers["PC1"],
    customers["PC2"],
    c=customers["cluster"],
    alpha=0.75,
)

plt.xlabel(f"PC1 ({pc1_var:.2f}% дисперсії)")
plt.ylabel(f"PC2 ({pc2_var:.2f}% дисперсії)")
plt.title("PCA-візуалізація кластерів клієнтів")
plt.grid(True)
plt.colorbar(scatter, label="Кластер")
plt.tight_layout()
plt.show()


# ============================================================
# 20. 3D-ГРАФІК RFM
# ============================================================

customers["log_recency"] = np.log1p(customers["recency_days"])
customers["log_frequency"] = np.log1p(customers["frequency_orders"])
customers["log_revenue"] = np.log1p(customers["revenue"])

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")

scatter_3d = ax.scatter(
    customers["log_recency"],
    customers["log_frequency"],
    customers["log_revenue"],
    c=customers["cluster"],
    s=45,
    alpha=0.8,
)

ax.set_xlabel("log(1 + Recency)")
ax.set_ylabel("log(1 + Frequency)")
ax.set_zlabel("log(1 + Revenue)")
ax.set_title("3D-візуалізація кластерів клієнтів за RFM-ознаками")

legend = ax.legend(*scatter_3d.legend_elements(), title="Кластери")
ax.add_artist(legend)

plt.tight_layout()
plt.show()


# ============================================================
# 21. ГРАФІКИ ПРОФІЛЮ КЛАСТЕРІВ
# ============================================================

for col in [
    "recency_days",
    "frequency_orders",
    "revenue",
    "avg_order_value",
    "unique_products",
]:
    plt.figure(figsize=(8, 5))
    cluster_profile[col].plot(kind="bar")
    plt.xlabel("Кластер")
    plt.ylabel(col)
    plt.title(f"Середнє значення {col} за кластерами")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.show()


# ============================================================
# 22. ТАБЛИЦЯ СЕГМЕНТІВ ДЛЯ ОПТИМІЗАЦІЙНОЇ МОДЕЛІ
# ============================================================

segments_for_model = cluster_profile.reset_index().rename(
    columns={
        "cluster": "segment_id",
        "recency_days": "avg_recency",
        "frequency_orders": "avg_frequency",
        "revenue": "avg_monetary",
        "avg_order_value": "avg_check",
        "unique_products": "avg_product_diversity",
        "quantity_total": "avg_quantity_total",
        "quantity_per_order": "avg_quantity_per_order",
        "revenue_per_product": "avg_revenue_per_product",
    }
)

segments_for_model = segments_for_model[
    [
        "segment_id",
        "segment_name",
        "customers_count",
        "customers_share_percent",
        "avg_recency",
        "avg_frequency",
        "avg_monetary",
        "avg_check",
        "avg_product_diversity",
        "avg_quantity_total",
        "avg_quantity_per_order",
        "avg_revenue_per_product",
        "segment_attractiveness",
        "budget_priority",
        "marketing_recommendation",
    ]
]

max_monetary = segments_for_model["avg_monetary"].max()
max_attractiveness = segments_for_model["segment_attractiveness"].max()

segments_for_model["economic_coef"] = (
    0.8 + 0.6 * segments_for_model["avg_monetary"] / max_monetary
).round(3)

segments_for_model["marketing_fit_coef"] = (
    0.9 + 0.3 * segments_for_model["segment_attractiveness"] / max_attractiveness
).round(3)


# ============================================================
# 23. ІНФОРМАЦІЯ ПРО PCA
# ============================================================

pca_info = pd.DataFrame(
    {
        "Показник": ["PC1", "PC2", "Разом"],
        "Пояснена дисперсія, %": [
            round(pc1_var, 2),
            round(pc2_var, 2),
            round(total_var, 2),
        ],
    }
)


explanation_df = pd.DataFrame(
    {
        "Етап": [
            "1. Вихідні дані",
            "2. Ознаки кластеризації",
            "3. Вибір кількості кластерів",
            "4. Кластеризація",
            "5. Сегментація",
            "6. PCA",
            "7. Використання результатів",
        ],
        "Пояснення": [
            "Для аналізу використано дані клієнтів інтернет-магазину з показниками купівельної поведінки.",
            "Кластеризацію виконано за RFM-ознаками: recency_days, frequency_orders, revenue.",
            f"Кількість кластерів визначено автоматично методом ліктя. Обране значення k = {optimal_k}.",
            "Метод K-Means поділив клієнтів на однорідні групи за схожістю купівельної поведінки.",
            "Отримані кластери інтерпретовано як маркетингові сегменти клієнтів.",
            f"Метод PCA використано для візуалізації кластерів. Перші дві компоненти пояснюють {round(total_var, 2)}% дисперсії.",
            "Таблиця segments_for_model може бути використана як вхідна інформація для оптимізаційної моделі маркетингового бюджету.",
        ],
    }
)

customers_output = customers[
    [
        "customer_id",
        "frequency_orders",
        "quantity_total",
        "revenue",
        "avg_order_value",
        "recency_days",
        "unique_products",
        "quantity_per_order",
        "revenue_per_product",
        "cluster",
        "segment_name",
        "PC1",
        "PC2",
    ]
].copy()

try:
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        customers_output.to_excel(writer, sheet_name="Клієнти_кластери", index=False)
        cluster_profile_output.to_excel(writer, sheet_name="Профіль_кластерів", index=False)
        segments_for_model.to_excel(writer, sheet_name="Сегменти_для_моделі", index=False)
        k_results.to_excel(writer, sheet_name="Оцінка_k", index=False)
        pca_info.to_excel(writer, sheet_name="PCA", index=False)
        explanation_df.to_excel(writer, sheet_name="Пояснення", index=False)

except PermissionError:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"K_means_results_{timestamp}.xlsx")

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        customers_output.to_excel(writer, sheet_name="Клієнти_кластери", index=False)
        cluster_profile_output.to_excel(writer, sheet_name="Профіль_кластерів", index=False)
        segments_for_model.to_excel(writer, sheet_name="Сегменти_для_моделі", index=False)
        k_results.to_excel(writer, sheet_name="Оцінка_k", index=False)
        pca_info.to_excel(writer, sheet_name="PCA", index=False)
        explanation_df.to_excel(writer, sheet_name="Пояснення", index=False)


print("\n" + "=" * 70)
print("КЛАСТЕРИЗАЦІЮ ТА СЕГМЕНТАЦІЮ ЗАВЕРШЕНО")
print("=" * 70)

print("\nВикористані ознаки:")
print(", ".join(features))

print("\nКількість кластерів визначена автоматично методом ліктя:")
print(optimal_k)

print("\nSilhouette score для обраної кількості кластерів:")
print(round(chosen_silhouette, 4))

print("\nPCA разом пояснює:")
print(round(total_var, 2), "% дисперсії")

print("\nExcel-файл з усіма результатами збережено тут:")
print(output_file)

print("\nУ файлі є листи:")
print("- Клієнти_кластери")
print("- Профіль_кластерів")
print("- Сегменти_для_моделі")
print("- Оцінка_k")
print("- PCA")
print("- Пояснення")
