# Chatbot de Sabentis Basado en Recuperación

---

## Índice

1. [Introducción](#introducción)
2. [Justificación del Proyecto](#justificación-del-proyecto)
3. [Objetivos](#objetivos)
4. [Arquitectura](#arquitectura)
5. [Librerías y Herramientas Utilizadas](#librerías-y-herramientas-utilizadas)
6. [Metodología](#metodología)
7. [Equipo](#equipo)
8. [Licencia](#licencia)

---

## Introducción

Este proyecto consiste en el desarrollo de un chatbot avanzado para la plataforma Sabentis, diseñado para optimizar la gestión de la Seguridad y Salud en el Trabajo (SST). El chatbot proporciona a los usuarios asistencia en tiempo real y acceso eficiente a la información contenida en más de 100 manuales relacionados con los 43 módulos que ofrece Sabentis.

## Justificación del Proyecto

El desarrollo del chatbot de Sabentis aborda la necesidad crítica de optimizar la gestión de SST. La extensa documentación y estructura modular de Sabentis requieren un medio eficiente de interacción con el usuario y recuperación de información, lo cual se maneja de manera efectiva con la integración de un chatbot.

## Objetivos

1. **Mejorar la Eficiencia Operativa:** Facilitar el acceso rápido y preciso a la información sobre SST.
2. **Automatizar Respuestas y Gestiones:** Implementar un sistema de respuestas automáticas a consultas frecuentes.
3. **Reducir Riesgos Laborales:** Mejorar la comunicación del usuario con el sistema de gestión de SST para una identificación y gestión de riesgos más rápida.
4. **Incrementar la Satisfacción del Cliente:** Proporcionar respuestas rápidas y personalizadas para mejorar la experiencia del usuario.
5. **Impacto a Nivel de Negocio:** Aumentar la competitividad de Sabentis mediante la adopción de tecnologías avanzadas.

### Requisitos Previos

- Python 3.8 o superior
- Docker
- Git

## Arquitectura

El chatbot utiliza un enfoque de Generación Aumentada por Recuperación (RAG), combinando técnicas de recuperación con modelos de generación de lenguaje para proporcionar respuestas precisas y contextualmente relevantes. El sistema está diseñado para manejar la documentación dinámica y extensa proporcionada por Sabentis.

### Componentes Clave

- **Preprocesamiento de Datos:** Utilizando PyMuPDF para la extracción y limpieza de texto.
- **Modelos de Incrustación:** Implementando modelos como BERT, Word2Vec, TF-IDF y ADA de OpenAI.
- **Mecanismo de Recuperación:** Utilizando la similitud de coseno para una recuperación eficiente.
- **Interfaz de Usuario:** Una UI simple e interactiva construida con Flask.

## Librerías y Herramientas Utilizadas

- **Python:** Lenguaje de programación principal.
- **PyMuPDF:** Para la extracción y procesamiento de texto de documentos PDF.
- **NLTK:** Para tareas de procesamiento de lenguaje natural.
- **SpaCy:** Biblioteca avanzada para el procesamiento de lenguaje natural.
- **Hugging Face Transformers:** Para modelos de lenguaje preentrenados.
- **Docker:** Para la creación de contenedores.
- **Trello:** Para la gestión del proyecto y la planificación de sprints.

## Metodología

El proyecto sigue la metodología SCRUM, con sprints de 2-3 semanas. El desarrollo se dividió en dos fases principales:

1. **Configuración Inicial:** Creación de un pipeline funcional, incluyendo el procesamiento de datos y la UI.
2. **Integración de Modelos:** Implementación y ajuste fino de varios modelos de incrustación y recuperación.

## Equipo

- **Brandon Maldonado Alonso**
- **Victor Aranda Belmonte**
- **Verónica Sánchez Muñoz**

### Supervisores

- **Tetiana Klymchuk**
- **Albert Puntí**


## Licencia

Este proyecto está licenciado bajo la Licencia MIT. Consulta el archivo LICENSE para más detalles.
