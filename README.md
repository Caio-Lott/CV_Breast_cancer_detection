
# 🧠 Classificação de Câncer de Mama com Redes Neurais Convolucionais (CNN)

Este repositório apresenta uma abordagem de aprendizado profundo para **classificação automática de imagens histopatológicas** de câncer de mama utilizando uma **Rede Neural Convolucional (CNN)**. A base utilizada é a **BreakHis**, contendo imagens em diferentes ampliações de tumores benignos e malignos.

---

## 📊 Base de Dados — BreakHis

A base de dados **Breast Cancer Histopathological Image Classification (BreakHis)** é composta por 9.109 imagens microscópicas de tecidos tumorais de mama (700x460 pixels, RGB), coletadas de 82 pacientes com ampliações de 40X, 100X, 200X e 400X.

🔗 Acesse: [https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/](https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/)

Neste projeto, utilizamos apenas imagens de ampliação **400X**.

### 🧬 Grupos da Base
- **Benigno**: Tumores não invasivos, crescimento lento, sem capacidade de metástase.
- **Maligno**: Tumores invasivos com potencial de destruição de tecidos e metástase (sinônimo de câncer).

---

## ⚙️ Como Executar

### 1. Clone o repositório

```bash
git clone https://github.com/seu-usuario/seu-repositorio.git
cd seu-repositorio
```

### 2. Crie um ambiente virtual e instale as dependências

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

pip install -r requirements.txt
```

### 3. Baixe a base de dados

1. Acesse o site oficial: [BreakHis](https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/)
2. Faça o download das imagens.
3. Extraia os arquivos em uma pasta `dataset/` na raiz do projeto com esta estrutura:

```
dataset/
├── benign/
│   └── (imagens benignas)
├── malignant/
│   └── (imagens malignas)
```

---

### 4. Execute o script principal

```bash
python main.py
```

O script:
- Carrega as imagens da base
- Realiza o pré-processamento
- Treina uma CNN simples para classificação
- Exibe e salva métricas e imagens de exemplo segmentadas

---

## 🧠 Tecnologias Utilizadas

- Python 3.10+
- TensorFlow / Keras
- OpenCV / PIL
- Matplotlib
- Scikit-learn

---

## 📌 Observações

> ⚠️ Devido à natureza colorida e detalhada das imagens histológicas, foi necessário **limitar o pré-processamento** para evitar perda de informações relevantes.  
> Imagens excessivamente tratadas podem comprometer a performance da CNN.

---

## 👨‍💻 Autor

Caio Lott  
Projeto desenvolvido para fins acadêmicos, utilizando técnicas de visão computacional aplicadas à área da saúde.

📬 Contato: [LinkedIn](https://www.linkedin.com/in/Caio-Lott)