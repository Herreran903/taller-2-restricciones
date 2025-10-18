# Taller 2 — Extensiones del Job Shop Scheduling Problem en MiniZinc

**Equipo:** John Freddy Belalcázar · Samuel Galindo Cuevas · Nicolás Herrera Marulanda  
**Asignatura:** Programación por Restricciones — Escuela de Ingeniería de Sistemas y Computación, Universidad del Valle  
**Profesor:** Robinson Andrey Duque Agudelo  
**Fecha:** Octubre de 2025  

---

## Descripción general

Este repositorio contiene la implementación de **dos variaciones del problema Job Shop Scheduling Problem (JSSP)** desarrolladas en el marco del **Taller 2 de Programación por Restricciones**.

Cada modelo se formula e implementa en **MiniZinc**, extendiendo el JSSP clásico con nuevas restricciones u objetivos que reflejan escenarios industriales reales:

1. **Job Shop con mantenimiento programado**  
   Cada máquina presenta intervalos de inactividad (mantenimiento preventivo) durante los cuales no puede ejecutar operaciones.

2. **Job Shop con prioridades y fechas límite**  
   Cada trabajo tiene una prioridad y una fecha límite. Se minimiza la suma ponderada de tardanzas \( \sum w_i \cdot \max(0, \text{end}_i - \text{due}_i) \).

Ambos modelos fueron evaluados bajo distintas estrategias de búsqueda y solvers (Gecode / Chuffed), generando 10 instancias de prueba por variante y un **informe LaTeX** con resultados, análisis y conclusiones.

---

## Estructura del repositorio


```
taller-2-restricciones/
├─ main.tex
├─ preambulo.tex
├─ README.md
├─ .gitignore .gitattributes .editorconfig Makefile
│
├─ refs/
│ ├─ john.bib
│ ├─ samuel.bib
│ └─ nicolas.bib
│
├─ secciones/
│ ├─ 01-jobshop_mantenimiento/
│ │ ├─ 00-jobshop_mantenimiento-intro.tex
│ │ ├─ 01-modelo.tex
│ │ ├─ 02-implementacion.tex
│ │ ├─ 03-pruebas.tex
│ │ ├─ 04-arboles.tex
│ │ └─ 05-analisis-y-conclusiones.tex
│ └─ 02-jobshop_tardanza/
│ ├─ 00-jobshop_tardanza-intro.tex
│ ├─ 01-modelo.tex
│ ├─ 02-implementacion.tex
│ ├─ 03-pruebas.tex
│ ├─ 04-arboles.tex
│ └─ 05-analisis-y-conclusiones.tex
│
├─ modelos/
│ ├─ jobshop_mantenimiento/
│ │ ├─ jobshop_mantenimiento.mzn
│ │ └─ tests/ # 10 instancias .dzn
│ └─ jobshop_tardanza/
│ ├─ jobshop_tardanza.mzn
│ └─ tests/
│
└─ script/
└─ run_experiments.py # Ejecutor automático de MiniZinc
```

> **Nota:** cada subcarpeta de `modelos/` contiene su modelo principal, las instancias `.dzn` y los resultados generados automáticamente.

---

## Ejecución de los modelos

Los modelos pueden ejecutarse manualmente desde MiniZinc IDE o desde la consola.

### 1. MiniZinc IDE
1. Abrir el archivo `.mzn` correspondiente.  
2. Asociar un archivo `.dzn` desde *Data*.  
3. Seleccionar el solver (`Chuffed` o `Gecode`).  
4. Ejecutar (*Run* o `Ctrl + R`).  
5. Revisar el `end`, el makespan y la matriz de tiempos de inicio.

### 2. Consola (CLI)

**macOS / Linux**
```bash
minizinc --solver Chuffed modelos/jobshop_mantenimiento/jobshop_mantenimiento.mzn modelos/jobshop_mantenimiento/tests/inst_01.dzn
```

**Windows (PowerShell)**
```powershell
minizinc --solver Gecode modelos\jobshop_tardanza\jobshop_tardanza.mzn modelos\jobshop_tardanza\tests\inst_01.dzn
```

> También puede añadirse la opción `--statistics` para obtener datos de búsqueda y fallos:
> ```bash
> minizinc --solver Chuffed --statistics modelos/jobshop_tardanza/jobshop_tardanza.mzn modelos/jobshop_tardanza/tests/inst_05.dzn
> ```

---

## Ejecución automatizada con el script

El archivo `script/run_experiments.py` automatiza la ejecución masiva de modelos sobre múltiples instancias, estrategias y solvers.

### Requisitos

- **MiniZinc** instalado y disponible en el `PATH`.
- **Python 3.x**.

#### Añadir MiniZinc al `PATH`

**macOS (MiniZinc IDE)**
```bash
export PATH="/Applications/MiniZincIDE.app/Contents/Resources:$PATH"
```

**Windows (PowerShell)**
```powershell
$env:PATH = "C:\Program Files\MiniZinc\;" + $env:PATH
```

## Ejemplos por problema

### Job Shop con mantenimiento
```bash
python3 script/run_experiments.py \
  --base-dir modelos/jobshop_mantenimiento \
  --model jobshop_mantenimiento.mzn \
  --data-dir tests \
  --solver gecode chuffed \
  --strategy ff_min wdeg_split inorder_min \
  --time-limit 60000
```

### Job Shop con tardanza ponderada
```bash
python3 script/run_experiments.py \
  --base-dir modelos/jobshop_tardanza \
  --model jobshop_tardanza.mzn \
  --data-dir tests \
  --solver gecode chuffed \
  --strategy ff_min wdeg_split inorder_min \
  --time-limit 60000
```