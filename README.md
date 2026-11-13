# Producción fotovoltaica: física + IA y salud de inversores

**Dashboard interactivo:** `docs/index.html` (publicado con GitHub Pages)

Dos cosas que pide cualquier operador de una planta solar:

1. **¿Cuánto debería estar produciendo?** Modelo de producción esperada a partir de irradiancia y temperatura, comparando un modelo **físico**, uno de **machine learning** y uno **híbrido** (física + ML sobre el residuo).
2. **¿Qué equipos están perdiendo energía?** Detección de inversores con bajo rendimiento comparando cada uno con sus vecinos en el mismo instante, y estimación de la energía y el dinero que se pierden.

Proyecto demostrativo con un dataset público (planta de 22 inversores, 34 días, 15 minutos). No usa datos de clientes.

## Resultados (10 días de test)

| Modelo | nRMSE (24 días de entrenamiento) | nRMSE (solo 5 días) |
|---|---|---|
| Físico (1 parámetro) | 2,7 % | 2,7 % |
| Machine learning | 2,4 % | 3,2 % |
| **Híbrido** | **2,3 %** | **2,6 %** |

- Con datos suficientes, el híbrido es el más preciso y el que menos se desvía en energía total.
- Con poco histórico (una planta recién conectada), el ML puro empeora y el híbrido se mantiene: la física aporta lo que faltan datos.

**Salud de inversores:** 2 de 22 inversores producen un 8 % y un 7 % menos que la mediana de la planta de forma sostenida. Extrapolado a un año, unos 505 MWh, **≈ 30.000 €/año** con un precio supuesto de 60 €/MWh.

> El modelo usa irradiancia medida, así que estima la producción esperada (útil para detectar pérdidas). Alimentado con previsión meteorológica sirve para prever producción, con un error mayor que depende de la calidad de esa previsión.

## Cómo ejecutarlo

```bash
pip install -r requirements.txt
python src/train.py          # modelos, métricas y salud de inversores -> results/results.json
python shared/build.py .     # regenera docs/index.html
```

## Cómo se aplicaría en una planta

- Datos de SCADA (inversores, string boxes) y estación meteorológica de la planta.
- Línea base de producción esperada por equipo y alertas cuando un inversor o string se desvía.
- Cuantificación de pérdidas en MWh y € para priorizar intervenciones de O&M.

---

**Álvaro Martín Coll** · Físico · Ingeniero de IA para industria y energía · [Perfil en Malt](https://www.malt.com/profile/alvaromartincoll)

Datos: *Solar Power Generation Data* (A. Kannal), publicado en Kaggle; planta 1.
