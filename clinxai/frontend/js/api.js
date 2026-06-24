const API_BASE = "http://127.0.0.1:8000";

const API = {
    async predictFused(formData, params) {
        // params: {age, gender, ...}
        const queryString = new URLSearchParams(params).toString();
        const response = await fetch(`${API_BASE}/predict/fused?${queryString}`, {
            method: 'POST',
            body: formData // contains 'file'
        });
        if (!response.ok) throw new Error("Prediction failed");
        return await response.json();
    },

    async explainImage(formData) {
        const response = await fetch(`${API_BASE}/explain/image`, {
            method: 'POST',
            body: formData
        });
        if (!response.ok) throw new Error("Image explanation failed");
        return await response.json();
    },

    async explainPatient(itemJson) {
        const response = await fetch(`${API_BASE}/explain/patient`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(itemJson)
        });
        if (!response.ok) throw new Error("Patient explanation failed");
        return await response.json();
    }
};
