"""Local feature-contribution explanations for the linear router."""

from __future__ import annotations

from sklearn.pipeline import Pipeline

from src.data import clean_ticket_text


def explain_prediction(
    model: Pipeline,
    text: str,
    predicted_department: str,
    top_n: int = 8,
) -> list[dict[str, float | str]]:
    """Return the strongest positive TF-IDF contributions for one prediction."""

    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["classifier"]
    transformed = vectorizer.transform([clean_ticket_text(text)])
    class_index = list(classifier.classes_).index(predicted_department)
    coefficients = classifier.coef_[class_index]
    feature_names = vectorizer.get_feature_names_out()

    row = transformed.getrow(0)
    contributions = row.data * coefficients[row.indices]
    ranked = sorted(
        zip(row.indices, contributions, strict=True),
        key=lambda item: item[1],
        reverse=True,
    )

    return [
        {"feature": str(feature_names[index]), "contribution": float(score)}
        for index, score in ranked
        if score > 0
    ][:top_n]

