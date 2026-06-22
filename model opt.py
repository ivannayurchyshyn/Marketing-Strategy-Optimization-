import os
from datetime import datetime
import numpy as np
import pandas as pd

try:
    import pulp
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Не встановлено бібліотеку PuLP. Встанови її командою:\n"
        "pip install pulp"
    ) from exc



input_file = r"C:\Users\Ivann\OneDrive - lnu.edu.ua\4курс\Курсова\КОД\K_means_results.xlsx"

if not os.path.exists(input_file):
    raise FileNotFoundError(f"Файл не знайдено: {input_file}")

output_dir = os.path.dirname(input_file)
output_file = os.path.join(output_dir, "marketing_optimization_results.xlsx")

print("=" * 70)
print("ОПТИМІЗАЦІЯ МАРКЕТИНГОВОГО БЮДЖЕТУ")
print("=" * 70)
print("\nВхідний файл:")
print(input_file)



segments = pd.read_excel(input_file, sheet_name="Сегменти_для_моделі")

required_columns = [
    "segment_id",
    "segment_name",
    "customers_count",
    "customers_share_percent",
    "avg_recency",
    "avg_frequency",
    "avg_monetary",
    "avg_check",
    "avg_product_diversity",
    "segment_attractiveness",
    "economic_coef",
    "marketing_fit_coef",
]

missing = [col for col in required_columns if col not in segments.columns]
if missing:
    raise ValueError(
        "У листі 'Сегменти_для_моделі' не знайдено колонки: "
        + ", ".join(missing)
    )

segments = segments[required_columns].copy()

for col in [
    "segment_id",
    "customers_count",
    "customers_share_percent",
    "avg_recency",
    "avg_frequency",
    "avg_monetary",
    "avg_check",
    "avg_product_diversity",
    "segment_attractiveness",
    "economic_coef",
    "marketing_fit_coef",
]:
    segments[col] = pd.to_numeric(segments[col], errors="coerce")

segments = segments.dropna()

if len(segments) == 0:
    raise ValueError("Після очищення таблиця сегментів порожня.")

segments["segment_id"] = segments["segment_id"].astype(int)


min_att = segments["segment_attractiveness"].min()
max_att = segments["segment_attractiveness"].max()

if max_att == min_att:
    segments["attractiveness_norm"] = 0.5
else:
    segments["attractiveness_norm"] = (
        (segments["segment_attractiveness"] - min_att) /
        (max_att - min_att)
    )

print("\nЗавантажено сегменти:")
print(segments[["segment_id", "segment_name", "customers_count", "segment_attractiveness"]])


TOTAL_BUDGET = 50000

MAX_RISK = 12000

MIN_MARKETING_EFFECT = 22000

MIN_SELECTED_SEGMENTS = min(2, len(segments))
MAX_SELECTED_SEGMENTS = len(segments)

MIN_SELECTED_PRODUCTS = 2
MAX_SELECTED_PRODUCTS = 4

MIN_USED_CHANNELS = 2
MAX_USED_CHANNELS = 5

channels = pd.DataFrame([
    {
        "channel_id": 1,
        "channel_name": "Email-маркетинг",
        "base_return": 2.30,
        "marketing_effect_coef": 1.05,
        "risk_coef": 0.12,
        "fixed_cost": 300,
        "min_spend": 500,
        "max_spend": 10000,
    },
    {
        "channel_id": 2,
        "channel_name": "Search Ads",
        "base_return": 2.05,
        "marketing_effect_coef": 1.20,
        "risk_coef": 0.20,
        "fixed_cost": 900,
        "min_spend": 1000,
        "max_spend": 15000,
    },
    {
        "channel_id": 3,
        "channel_name": "Social Media Ads",
        "base_return": 1.85,
        "marketing_effect_coef": 1.30,
        "risk_coef": 0.28,
        "fixed_cost": 800,
        "min_spend": 1000,
        "max_spend": 13000,
    },
    {
        "channel_id": 4,
        "channel_name": "Discount Campaigns",
        "base_return": 1.55,
        "marketing_effect_coef": 0.95,
        "risk_coef": 0.18,
        "fixed_cost": 500,
        "min_spend": 700,
        "max_spend": 9000,
    },
    {
        "channel_id": 5,
        "channel_name": "Marketplace Promotion",
        "base_return": 1.75,
        "marketing_effect_coef": 1.10,
        "risk_coef": 0.22,
        "fixed_cost": 700,
        "min_spend": 800,
        "max_spend": 12000,
    },
])


product_groups = pd.DataFrame([
    {
        "product_id": 1,
        "product_group": "Home Decor",
        "product_return_coef": 1.10,
        "marketing_effect_coef": 1.05,
        "risk_coef": 0.18,
        "min_budget": 2000,
        "max_budget": 22000,
    },
    {
        "product_id": 2,
        "product_group": "Kitchen & Tableware",
        "product_return_coef": 1.00,
        "marketing_effect_coef": 1.00,
        "risk_coef": 0.16,
        "min_budget": 2000,
        "max_budget": 20000,
    },
    {
        "product_id": 3,
        "product_group": "Gifts & Sets",
        "product_return_coef": 1.20,
        "marketing_effect_coef": 1.15,
        "risk_coef": 0.20,
        "min_budget": 2000,
        "max_budget": 24000,
    },
    {
        "product_id": 4,
        "product_group": "Seasonal Products",
        "product_return_coef": 1.25,
        "marketing_effect_coef": 1.25,
        "risk_coef": 0.30,
        "min_budget": 2000,
        "max_budget": 18000,
    },
])

periods = pd.DataFrame([
    {"period_id": 1, "period_name": "I квартал", "seasonality_coef": 0.95, "period_budget": 12500},
    {"period_id": 2, "period_name": "II квартал", "seasonality_coef": 1.00, "period_budget": 12500},
    {"period_id": 3, "period_name": "III квартал", "seasonality_coef": 1.05, "period_budget": 12500},
    {"period_id": 4, "period_name": "IV квартал", "seasonality_coef": 1.20, "period_budget": 12500},
])

scenarios = pd.DataFrame([
    {"scenario_id": 1, "scenario_name": "Песимістичний", "probability": 0.25, "effect_multiplier": 0.85, "risk_multiplier": 1.20},
    {"scenario_id": 2, "scenario_name": "Базовий", "probability": 0.50, "effect_multiplier": 1.00, "risk_multiplier": 1.00},
    {"scenario_id": 3, "scenario_name": "Оптимістичний", "probability": 0.25, "effect_multiplier": 1.15, "risk_multiplier": 0.90},
])

expected_effect_multiplier = float(
    (scenarios["probability"] * scenarios["effect_multiplier"]).sum()
)

expected_risk_multiplier = float(
    (scenarios["probability"] * scenarios["risk_multiplier"]).sum()
)

parameter_rows = []

for _, ch in channels.iterrows():
    for _, seg in segments.iterrows():
        for _, prod in product_groups.iterrows():
            for _, per in periods.iterrows():

                # Очікуваний валовий ефект від 1 грн витрат.
                gross_return_coef = (
                    ch["base_return"]
                    * seg["economic_coef"]
                    * seg["marketing_fit_coef"]
                    * prod["product_return_coef"]
                    * per["seasonality_coef"]
                    * expected_effect_multiplier
                )

                # Маркетинговий ефект.
                marketing_effect_coef = (
                    ch["marketing_effect_coef"]
                    * seg["marketing_fit_coef"]
                    * prod["marketing_effect_coef"]
                    * per["seasonality_coef"]
                )

                # Ризик. Менш привабливі сегменти мають вищий ризик.
                segment_risk_factor = 1.25 - 0.35 * seg["attractiveness_norm"]

                risk_coef = (
                    ch["risk_coef"]
                    * prod["risk_coef"]
                    * segment_risk_factor
                    * expected_risk_multiplier
                )

                parameter_rows.append({
                    "channel_id": int(ch["channel_id"]),
                    "channel_name": ch["channel_name"],
                    "segment_id": int(seg["segment_id"]),
                    "segment_name": seg["segment_name"],
                    "product_id": int(prod["product_id"]),
                    "product_group": prod["product_group"],
                    "period_id": int(per["period_id"]),
                    "period_name": per["period_name"],
                    "gross_return_coef": round(float(gross_return_coef), 4),
                    "net_return_coef": round(float(gross_return_coef - 1), 4),
                    "marketing_effect_coef": round(float(marketing_effect_coef), 4),
                    "risk_coef": round(float(risk_coef), 4),
                })

params = pd.DataFrame(parameter_rows)


# ПОБУДОВА MILP-МОДЕЛІ


model = pulp.LpProblem("Marketing_Budget_Optimization", pulp.LpMaximize)

C = channels["channel_id"].astype(int).tolist()
S = segments["segment_id"].astype(int).tolist()
G = product_groups["product_id"].astype(int).tolist()
T = periods["period_id"].astype(int).tolist()

# Змінні витрат x[c,s,g,t]
x_vars = {}

for _, row in params.iterrows():
    key = (
        int(row["channel_id"]),
        int(row["segment_id"]),
        int(row["product_id"]),
        int(row["period_id"]),
    )
    x_vars[key] = pulp.LpVariable(
        f"x_c{key[0]}_s{key[1]}_g{key[2]}_t{key[3]}",
        lowBound=0,
        cat="Continuous",
    )

# Бінарна змінна активації каналу у періоді
y_channel_period = {
    (c, t): pulp.LpVariable(f"y_c{c}_t{t}", cat="Binary")
    for c in C
    for t in T
}

# Бінарна змінна використання каналу протягом усього періоду
u_channel = {
    c: pulp.LpVariable(f"u_c{c}", cat="Binary")
    for c in C
}

# Бінарна змінна вибору сегмента
z_segment = {
    s: pulp.LpVariable(f"z_s{s}", cat="Binary")
    for s in S
}

# Бінарна змінна вибору товарної групи
w_product = {
    g: pulp.LpVariable(f"w_g{g}", cat="Binary")
    for g in G
}


param_lookup = {
    (
        int(row["channel_id"]),
        int(row["segment_id"]),
        int(row["product_id"]),
        int(row["period_id"]),
    ): row
    for _, row in params.iterrows()
}

channel_lookup = {
    int(row["channel_id"]): row
    for _, row in channels.iterrows()
}

model += (
    pulp.lpSum(
        x_vars[key] * float(param_lookup[key]["net_return_coef"])
        for key in x_vars
    )
    -
    pulp.lpSum(
        y_channel_period[(c, t)] * float(channel_lookup[c]["fixed_cost"])
        for c in C
        for t in T
    )
), "Expected_Net_Result"



# Загальний бюджет.
model += (
    pulp.lpSum(x_vars.values())
    +
    pulp.lpSum(
        y_channel_period[(c, t)] * float(channel_lookup[c]["fixed_cost"])
        for c in C
        for t in T
    )
    <= TOTAL_BUDGET
), "Total_Budget"

# Бюджет за періодами.
period_budget_lookup = {
    int(row["period_id"]): float(row["period_budget"])
    for _, row in periods.iterrows()
}

for t in T:
    model += (
        pulp.lpSum(
            x_vars[(c, s, g, t)]
            for c in C
            for s in S
            for g in G
        )
        +
        pulp.lpSum(
            y_channel_period[(c, t)] * float(channel_lookup[c]["fixed_cost"])
            for c in C
        )
        <= period_budget_lookup[t]
    ), f"Period_Budget_t{t}"



for c in C:
    min_spend = float(channel_lookup[c]["min_spend"])
    max_spend = float(channel_lookup[c]["max_spend"])

    for t in T:
        spend_ct = pulp.lpSum(
            x_vars[(c, s, g, t)]
            for s in S
            for g in G
        )

        model += (
            spend_ct <= max_spend * y_channel_period[(c, t)]
        ), f"Max_Channel_Spend_c{c}_t{t}"

        model += (
            spend_ct >= min_spend * y_channel_period[(c, t)]
        ), f"Min_Channel_Spend_c{c}_t{t}"

        model += (
            y_channel_period[(c, t)] <= u_channel[c]
        ), f"Channel_Period_Link_c{c}_t{t}"

    model += (
        u_channel[c] <= pulp.lpSum(y_channel_period[(c, t)] for t in T)
    ), f"Channel_Use_Link_c{c}"

model += (
    pulp.lpSum(u_channel[c] for c in C) >= MIN_USED_CHANNELS
), "Min_Used_Channels"

model += (
    pulp.lpSum(u_channel[c] for c in C) <= MAX_USED_CHANNELS
), "Max_Used_Channels"



segment_budget_rows = []

for _, seg in segments.iterrows():
    s = int(seg["segment_id"])

    # Мінімальний бюджет для обраного сегмента.
    min_segment_budget = 1000

    # Верхня межа залежить від частки сегмента та привабливості.
    max_segment_budget = TOTAL_BUDGET * (
        0.10
        + 0.50 * float(seg["customers_share_percent"]) / 100
        + 0.20 * float(seg["attractiveness_norm"])
    )

    max_segment_budget = max(max_segment_budget, min_segment_budget + 1000)

    segment_budget_rows.append({
        "segment_id": s,
        "segment_name": seg["segment_name"],
        "min_segment_budget": round(min_segment_budget, 2),
        "max_segment_budget": round(max_segment_budget, 2),
    })

    spend_s = pulp.lpSum(
        x_vars[(c, s, g, t)]
        for c in C
        for g in G
        for t in T
    )

    model += (
        spend_s <= max_segment_budget * z_segment[s]
    ), f"Max_Segment_Spend_s{s}"

    model += (
        spend_s >= min_segment_budget * z_segment[s]
    ), f"Min_Segment_Spend_s{s}"

model += (
    pulp.lpSum(z_segment[s] for s in S) >= MIN_SELECTED_SEGMENTS
), "Min_Selected_Segments"

model += (
    pulp.lpSum(z_segment[s] for s in S) <= MAX_SELECTED_SEGMENTS
), "Max_Selected_Segments"

segment_budget_limits = pd.DataFrame(segment_budget_rows)



product_lookup = {
    int(row["product_id"]): row
    for _, row in product_groups.iterrows()
}

for g in G:
    min_budget = float(product_lookup[g]["min_budget"])
    max_budget = float(product_lookup[g]["max_budget"])

    spend_g = pulp.lpSum(
        x_vars[(c, s, g, t)]
        for c in C
        for s in S
        for t in T
    )

    model += (
        spend_g <= max_budget * w_product[g]
    ), f"Max_Product_Spend_g{g}"

    model += (
        spend_g >= min_budget * w_product[g]
    ), f"Min_Product_Spend_g{g}"

model += (
    pulp.lpSum(w_product[g] for g in G) >= MIN_SELECTED_PRODUCTS
), "Min_Selected_Products"

model += (
    pulp.lpSum(w_product[g] for g in G) <= MAX_SELECTED_PRODUCTS
), "Max_Selected_Products"



for c in C:
    for s in S:
        for g in G:
            for t in T:
                key = (c, s, g, t)

                # Якщо сегмент або товарна група не обрані,
                # витрати на них мають бути нульові.
                model += (
                    x_vars[key] <= TOTAL_BUDGET * z_segment[s]
                ), f"Link_X_Segment_c{c}_s{s}_g{g}_t{t}"

                model += (
                    x_vars[key] <= TOTAL_BUDGET * w_product[g]
                ), f"Link_X_Product_c{c}_s{s}_g{g}_t{t}"

                model += (
                    x_vars[key] <= TOTAL_BUDGET * y_channel_period[(c, t)]
                ), f"Link_X_Channel_c{c}_s{s}_g{g}_t{t}"


total_risk_expr = pulp.lpSum(
    x_vars[key] * float(param_lookup[key]["risk_coef"])
    for key in x_vars
)

total_marketing_effect_expr = pulp.lpSum(
    x_vars[key] * float(param_lookup[key]["marketing_effect_coef"])
    for key in x_vars
)

model += (
    total_risk_expr <= MAX_RISK
), "Max_Total_Risk"

model += (
    total_marketing_effect_expr >= MIN_MARKETING_EFFECT
), "Min_Marketing_Effect"



print("\nРозв'язання оптимізаційної моделі...")

solver = pulp.PULP_CBC_CMD(msg=False)
model.solve(solver)

status = pulp.LpStatus[model.status]

print("\nСтатус розв'язку:")
print(status)

if status != "Optimal":
    print("\nУвага: оптимальний розв'язок не знайдено.")
    print("Можливо, потрібно послабити обмеження бюджету, ризику або маркетингового ефекту.")



result_rows = []

for key, var in x_vars.items():
    spend = var.value()

    if spend is None:
        spend = 0

    if spend > 0.01:
        row = param_lookup[key]

        gross_return = spend * float(row["gross_return_coef"])
        net_result = spend * float(row["net_return_coef"])
        marketing_effect = spend * float(row["marketing_effect_coef"])
        risk_value = spend * float(row["risk_coef"])
        roas = gross_return / spend if spend > 0 else 0

        result_rows.append({
            "channel_id": key[0],
            "channel_name": row["channel_name"],
            "segment_id": key[1],
            "segment_name": row["segment_name"],
            "product_id": key[2],
            "product_group": row["product_group"],
            "period_id": key[3],
            "period_name": row["period_name"],
            "spend": round(spend, 2),
            "gross_return_coef": row["gross_return_coef"],
            "net_return_coef": row["net_return_coef"],
            "expected_gross_return": round(gross_return, 2),
            "expected_net_result": round(net_result, 2),
            "ROAS": round(roas, 3),
            "marketing_effect": round(marketing_effect, 2),
            "risk_value": round(risk_value, 2),
        })

results = pd.DataFrame(result_rows)

if len(results) == 0:
    results = pd.DataFrame(columns=[
        "channel_id",
        "channel_name",
        "segment_id",
        "segment_name",
        "product_id",
        "product_group",
        "period_id",
        "period_name",
        "spend",
        "expected_gross_return",
        "expected_net_result",
        "ROAS",
        "marketing_effect",
        "risk_value",
    ])


fixed_cost_total = sum(
    (y_channel_period[(c, t)].value() or 0) * float(channel_lookup[c]["fixed_cost"])
    for c in C
    for t in T
)

variable_spend_total = results["spend"].sum() if len(results) else 0
total_spend = variable_spend_total + fixed_cost_total
total_gross_return = results["expected_gross_return"].sum() if len(results) else 0
total_net_result = results["expected_net_result"].sum() - fixed_cost_total if len(results) else -fixed_cost_total
total_marketing_effect = results["marketing_effect"].sum() if len(results) else 0
total_risk = results["risk_value"].sum() if len(results) else 0

# ROAS показує, скільки очікуваного валового результату припадає
# на 1 одиницю змінних маркетингових витрат.
# Фіксовані витрати каналів окремо враховуються в чистому результаті.
total_roas = total_gross_return / variable_spend_total if variable_spend_total > 0 else 0

# Альтернативний ROAS з урахуванням фіксованих витрат.
total_roas_with_fixed_costs = total_gross_return / total_spend if total_spend > 0 else 0

summary = pd.DataFrame([
    {"indicator": "Статус розв'язку", "value": status},
    {"indicator": "Загальний бюджет", "value": TOTAL_BUDGET},
    {"indicator": "Змінні маркетингові витрати", "value": round(variable_spend_total, 2)},
    {"indicator": "Фіксовані витрати активації каналів", "value": round(fixed_cost_total, 2)},
    {"indicator": "Загальні витрати", "value": round(total_spend, 2)},
    {"indicator": "Очікуваний валовий результат", "value": round(total_gross_return, 2)},
    {"indicator": "Очікуваний чистий результат", "value": round(total_net_result, 2)},
    {"indicator": "ROAS", "value": round(total_roas, 3)},
    {"indicator": "ROAS з урахуванням фіксованих витрат", "value": round(total_roas_with_fixed_costs, 3)},
    {"indicator": "Сукупний маркетинговий ефект", "value": round(total_marketing_effect, 2)},
    {"indicator": "Мінімально необхідний маркетинговий ефект", "value": MIN_MARKETING_EFFECT},
    {"indicator": "Сукупний ризик", "value": round(total_risk, 2)},
    {"indicator": "Максимально допустимий ризик", "value": MAX_RISK},
])

if len(results):
    by_segment = (
        results.groupby(["segment_id", "segment_name"], as_index=False)
        .agg(
            spend=("spend", "sum"),
            expected_gross_return=("expected_gross_return", "sum"),
            expected_net_result=("expected_net_result", "sum"),
            marketing_effect=("marketing_effect", "sum"),
            risk_value=("risk_value", "sum"),
        )
        .round(2)
        .sort_values("spend", ascending=False)
    )
    by_segment["ROAS"] = (by_segment["expected_gross_return"] / by_segment["spend"]).round(3)

    by_channel = (
        results.groupby(["channel_id", "channel_name"], as_index=False)
        .agg(
            spend=("spend", "sum"),
            expected_gross_return=("expected_gross_return", "sum"),
            expected_net_result=("expected_net_result", "sum"),
            marketing_effect=("marketing_effect", "sum"),
            risk_value=("risk_value", "sum"),
        )
        .round(2)
        .sort_values("spend", ascending=False)
    )
    by_channel["ROAS"] = (by_channel["expected_gross_return"] / by_channel["spend"]).round(3)

    by_product = (
        results.groupby(["product_id", "product_group"], as_index=False)
        .agg(
            spend=("spend", "sum"),
            expected_gross_return=("expected_gross_return", "sum"),
            expected_net_result=("expected_net_result", "sum"),
            marketing_effect=("marketing_effect", "sum"),
            risk_value=("risk_value", "sum"),
        )
        .round(2)
        .sort_values("spend", ascending=False)
    )
    by_product["ROAS"] = (by_product["expected_gross_return"] / by_product["spend"]).round(3)

    by_period = (
        results.groupby(["period_id", "period_name"], as_index=False)
        .agg(
            spend=("spend", "sum"),
            expected_gross_return=("expected_gross_return", "sum"),
            expected_net_result=("expected_net_result", "sum"),
            marketing_effect=("marketing_effect", "sum"),
            risk_value=("risk_value", "sum"),
        )
        .round(2)
        .sort_values("period_id")
    )
    by_period["ROAS"] = (by_period["expected_gross_return"] / by_period["spend"]).round(3)
else:
    by_segment = pd.DataFrame()
    by_channel = pd.DataFrame()
    by_product = pd.DataFrame()
    by_period = pd.DataFrame()


activated_channels = []

for c in C:
    activated_channels.append({
        "channel_id": c,
        "channel_name": channel_lookup[c]["channel_name"],
        "used": int(round(u_channel[c].value() or 0)),
    })

activated_channels = pd.DataFrame(activated_channels)

selected_segments = pd.DataFrame([
    {
        "segment_id": s,
        "selected": int(round(z_segment[s].value() or 0)),
    }
    for s in S
]).merge(
    segments[["segment_id", "segment_name"]],
    on="segment_id",
    how="left",
)

selected_products = pd.DataFrame([
    {
        "product_id": g,
        "selected": int(round(w_product[g].value() or 0)),
    }
    for g in G
]).merge(
    product_groups[["product_id", "product_group"]],
    on="product_id",
    how="left",
)



explanation = pd.DataFrame([
    {
        "section": "Зміст етапу",
        "text": (
            "На цьому етапі результати кластеризації клієнтів використовуються "
            "як вхідні дані для оптимізаційної моделі маркетингової стратегії."
        ),
    },
    {
        "section": "Сегменти",
        "text": (
            "Кожен segment_id відповідає маркетинговому сегменту, отриманому "
            "на основі K-Means кластеризації та подальшої інтерпретації."
        ),
    },
    {
        "section": "Змінна x",
        "text": (
            "Змінна x показує обсяг маркетингових витрат, спрямованих через певний "
            "канал на певний сегмент, товарну групу та період."
        ),
    },
    {
        "section": "Бінарні змінні",
        "text": (
            "Бінарні змінні визначають, чи активовано канал, чи обрано сегмент "
            "і чи включено товарну групу до маркетингової програми."
        ),
    },
    {
        "section": "Цільова функція",
        "text": (
            "Цільова функція максимізує очікуваний чистий економічний результат "
            "з урахуванням змінних маркетингових витрат і фіксованих витрат активації каналів."
        ),
    },
    {
        "section": "Обмеження",
        "text": (
            "Модель враховує загальний бюджет, бюджети за періодами, межі витрат "
            "за каналами, вибір сегментів і товарних груп, допустимий ризик "
            "та мінімальний маркетинговий ефект."
        ),
    },
    {
        "section": "ROAS",
        "text": (
            "ROAS розраховується як відношення очікуваного валового результату "
            "до маркетингових витрат. У таблицях наведено ROAS для всього плану, "
            "а також окремо за сегментами, каналами, товарними групами та періодами."
        ),
    },
])



try:
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Підсумок", index=False)
        results.to_excel(writer, sheet_name="Оптимальний_план", index=False)
        by_segment.to_excel(writer, sheet_name="За_сегментами", index=False)
        by_channel.to_excel(writer, sheet_name="За_каналами", index=False)
        by_product.to_excel(writer, sheet_name="За_товарами", index=False)
        by_period.to_excel(writer, sheet_name="За_періодами", index=False)
        segments.to_excel(writer, sheet_name="Вхідні_сегменти", index=False)
        channels.to_excel(writer, sheet_name="Канали", index=False)
        product_groups.to_excel(writer, sheet_name="Товарні_групи", index=False)
        periods.to_excel(writer, sheet_name="Періоди", index=False)
        scenarios.to_excel(writer, sheet_name="Сценарії", index=False)
        params.to_excel(writer, sheet_name="Параметри", index=False)
        segment_budget_limits.to_excel(writer, sheet_name="Межі_сегментів", index=False)
        activated_channels.to_excel(writer, sheet_name="Активовані_канали", index=False)
        selected_segments.to_excel(writer, sheet_name="Обрані_сегменти", index=False)
        selected_products.to_excel(writer, sheet_name="Обрані_товари", index=False)
        explanation.to_excel(writer, sheet_name="Пояснення", index=False)

except PermissionError:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"marketing_optimization_results_{timestamp}.xlsx")

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Підсумок", index=False)
        results.to_excel(writer, sheet_name="Оптимальний_план", index=False)
        by_segment.to_excel(writer, sheet_name="За_сегментами", index=False)
        by_channel.to_excel(writer, sheet_name="За_каналами", index=False)
        by_product.to_excel(writer, sheet_name="За_товарами", index=False)
        by_period.to_excel(writer, sheet_name="За_періодами", index=False)
        segments.to_excel(writer, sheet_name="Вхідні_сегменти", index=False)
        channels.to_excel(writer, sheet_name="Канали", index=False)
        product_groups.to_excel(writer, sheet_name="Товарні_групи", index=False)
        periods.to_excel(writer, sheet_name="Періоди", index=False)
        scenarios.to_excel(writer, sheet_name="Сценарії", index=False)
        params.to_excel(writer, sheet_name="Параметри", index=False)
        segment_budget_limits.to_excel(writer, sheet_name="Межі_сегментів", index=False)
        activated_channels.to_excel(writer, sheet_name="Активовані_канали", index=False)
        selected_segments.to_excel(writer, sheet_name="Обрані_сегменти", index=False)
        selected_products.to_excel(writer, sheet_name="Обрані_товари", index=False)
        explanation.to_excel(writer, sheet_name="Пояснення", index=False)


print("\n" + "=" * 70)
print("ОПТИМІЗАЦІЮ ЗАВЕРШЕНО")
print("=" * 70)

print("\nСтатус:", status)
print("Загальний бюджет:", TOTAL_BUDGET)
print("Загальні витрати:", round(total_spend, 2))
print("Очікуваний чистий результат:", round(total_net_result, 2))
print("ROAS:", round(total_roas, 3))
print("ROAS з урахуванням фіксованих витрат:", round(total_roas_with_fixed_costs, 3))
print("Сукупний маркетинговий ефект:", round(total_marketing_effect, 2))
print("Сукупний ризик:", round(total_risk, 2))

print("\nРезультати збережено у файл:")
print(output_file)

if len(results):
    print("\nРозподіл бюджету за сегментами:")
    print(by_segment)
    print("\nРозподіл бюджету за каналами:")
    print(by_channel)
else:
    print("\nОптимальний план витрат порожній. Перевір обмеження моделі.")
