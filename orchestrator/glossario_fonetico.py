"""
Glossário de respelling fonético para termos que o Kokoro (pipeline pt-BR)
tende a pronunciar mal -- nomes próprios e termos técnicos em inglês comuns
no nicho de gameplay/cultura pop.

IMPORTANTE: isso NUNCA deve substituir o texto oficial do roteiro
(legenda, matching de âncora, revisão). É aplicado só numa cópia do
texto, exclusivamente antes da chamada ao Kokoro.

Convenção de entrada: chaves em minúsculas, sem acento adicional além do
necessário para a pronúncia -- a substituição no texto é case-insensitive
e preserva a capitalização original da primeira letra da ocorrência.
"""
import re

# Ordene por especificidade: termos compostos ANTES de termos simples que
# sejam substring deles (ex: "kaer morhen" antes de qualquer parte isolada),
# senão a substituição parcial quebra a mais completa.
GLOSSARIO: dict[str, str] = {
    "kaer morhen": "ker mórren",
    "parry": "pérri",
    "witcher": "uícher",
    "geralt": "guéralt",
    "yennefer": "iênefer",
    "ciri": "círi",
    # adicione conforme for ouvindo problemas novos -- é cumulativo
}

# Termos mais longos primeiro, para não deixar substituição parcial de um
# termo curto atrapalhar o casamento de um termo composto que o contém.
_TERMOS_ORDENADOS = sorted(GLOSSARIO.keys(), key=len, reverse=True)


def aplicar_respelling(texto: str) -> str:
    """
    Retorna uma NOVA string com os termos do glossário substituídos pela
    grafia fonética aproximada, preservando a capitalização da primeira
    letra de cada ocorrência (ex: "Witcher" -> "Uícher", "witcher" -> "uícher").

    Não modifica o texto original -- use sempre em uma variável separada,
    nunca sobrescrevendo texto_locucao.
    """
    resultado = texto

    for termo_original in _TERMOS_ORDENADOS:
        substituto = GLOSSARIO[termo_original]
        # \b não cobre bem espaços internos de termos compostos ("kaer morhen"),
        # então usamos lookaround simples em vez de \b nas bordas
        padrao = re.compile(
            r"(?<![a-zA-ZÀ-ÿ])" + re.escape(termo_original) + r"(?![a-zA-ZÀ-ÿ])",
            re.IGNORECASE,
        )

        def _substituir(match: re.Match) -> str:
            original = match.group(0)
            if original[0].isupper():
                return substituto[0].upper() + substituto[1:]
            return substituto

        resultado = padrao.sub(_substituir, resultado)

    return resultado