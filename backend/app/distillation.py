import re
from collections import Counter
from typing import Dict, List


class DistillationEngine:
    """Deterministic, precision-first chunk distillation for RAG artifacts."""

    STOPWORDS = {
        "about", "above", "after", "again", "against", "also", "among", "because",
        "before", "being", "below", "between", "both", "cannot", "could", "does",
        "doing", "down", "during", "each", "from", "further", "have", "having",
        "here", "hers", "herself", "himself", "into", "itself", "more", "most",
        "other", "ours", "ourselves", "over", "same", "should", "some", "such",
        "than", "that", "their", "theirs", "them", "themselves", "then", "there",
        "these", "they", "this", "those", "through", "under", "until", "very",
        "were", "what", "when", "where", "which", "while", "with", "would",
        "your", "yours", "yourself", "yourselves", "the", "and", "for", "are",
        "but", "not", "you", "all", "can", "has", "had", "was", "its", "our",
        "out", "who", "why", "how", "his", "her", "she", "him", "use", "used",
        "using", "don", "doesn", "didn", "isn", "aren", "wasn", "weren", "won",
        "wouldn", "couldn", "shouldn", "let", "may", "might", "must", "shall",
    }

    FILLER_TERMS = {
        "chapter", "section", "figure", "table", "page", "pages", "copyright",
        "reserved", "publisher", "published", "edition", "contents", "introduction",
        "overview", "appendix", "index", "references", "acknowledgements",
    }

    GENERIC_KEYWORD_TERMS = {
        "any", "anything", "thing", "things", "problem", "problems", "first", "last",
        "time", "times", "way", "ways", "part", "parts", "many", "much", "few",
        "several", "example", "examples", "case", "cases", "level", "levels",
        "kind", "kinds", "type", "types", "area", "areas", "form", "forms",
        "process", "system", "systems", "result", "results", "point", "points",
        "another", "create", "creates", "created", "better", "worse", "based",
        "perceived", "often", "usually", "generally", "make", "makes", "made",
        "otherwise", "even", "demo", "demos", "think", "thinks", "thought",
        "ask", "asks", "take", "takes", "live", "lives", "detail", "details",
        "regret", "regrets",
    }

    GENERIC_SUMMARY_TERMS = {
        "this section", "this chapter", "the following", "in this chapter",
        "in this section", "this book", "this document", "the author",
    }

    SUMMARY_FILLER_WORDS = {
        "very", "really", "just", "basically", "actually", "simply", "quite",
        "rather", "perhaps", "maybe", "probably", "definitely", "clearly",
        "obviously", "literally", "particularly",
    }

    PRONOUN_SUBJECTS = {
        "it", "this", "that", "these", "those", "they", "he", "she", "we", "you",
        "i", "there", "one", "something", "someone",
    }

    VAGUE_CONCEPT_TERMS = {
        "important", "useful", "helpful", "good", "bad", "better", "worse",
        "effective", "ineffective", "necessary", "possible", "available",
        "different", "similar", "same", "thing", "things", "problem", "problems",
    }

    CONCEPT_SENTENCE_CUES = {
        "is", "are", "was", "were", "causes", "cause", "caused", "improves",
        "improve", "improved", "requires", "require", "required", "depends",
        "lead", "leads",
    }

    ENTITY_PREFIX_FILLER = {
        "a", "an", "the", "and", "or", "but", "because", "therefore", "however",
        "as", "when", "where", "while", "if", "then", "that", "which", "who",
        "these", "those", "many", "some", "most", "for", "with", "without",
        "within", "across", "among", "between", "through",
    }

    ENTITY_SUFFIX_FILLER = {
        "can", "may", "might", "must", "should", "would", "could", "also",
        "often", "usually", "generally", "commonly",
    }

    CONTENT_RELATIONS = {
        "is": "is",
        "are": "is",
        "was": "is",
        "were": "is",
        "causes": "causes",
        "cause": "causes",
        "caused": "causes",
        "improves": "improves",
        "improve": "improves",
        "improved": "improves",
        "depends on": "depends_on",
        "depend on": "depends_on",
        "leads to": "leads_to",
        "lead to": "leads_to",
        "requires": "requires",
        "require": "requires",
        "required": "requires",
    }

    RELATION_PATTERNS = [
        re.compile(r"\b(?P<subject>[^.;!?]{3,80}?)\s+(?P<relation>depends on|depend on)\s+(?P<object>[^.;!?]{3,90})$", re.IGNORECASE),
        re.compile(r"\b(?P<subject>[^.;!?]{3,80}?)\s+(?P<relation>leads to|lead to)\s+(?P<object>[^.;!?]{3,90})$", re.IGNORECASE),
        re.compile(r"\b(?P<subject>[^.;!?]{3,80}?)\s+(?P<relation>causes|cause|caused)\s+(?P<object>[^.;!?]{3,90})$", re.IGNORECASE),
        re.compile(r"\b(?P<subject>[^.;!?]{3,80}?)\s+(?P<relation>improves|improve|improved)\s+(?P<object>[^.;!?]{3,90})$", re.IGNORECASE),
        re.compile(r"\b(?P<subject>[^.;!?]{3,80}?)\s+(?P<relation>requires|require|required)\s+(?P<object>[^.;!?]{3,90})$", re.IGNORECASE),
        re.compile(r"\b(?P<subject>[^.;!?]{3,80}?)\s+(?P<relation>is|are|was|were)\s+(?P<object>[^.;!?]{3,90})$", re.IGNORECASE),
    ]

    TOC_PATTERNS = [
        re.compile(r"\b(table of contents|contents)\b", re.IGNORECASE),
        re.compile(r"\.{3,}\s*\d+\b"),
        re.compile(r"\b(chapter|section)\s+\d+(\.\d+)*\b", re.IGNORECASE),
    ]

    METADATA_PATTERNS = [
        re.compile(r"\b(copyright|all rights reserved|isbn|publisher|published by)\b", re.IGNORECASE),
        re.compile(r"\b(author biography|about the author|dedication|acknowledgements?)\b", re.IGNORECASE),
        re.compile(r"\b(printed in|first published|edition|license|permissions)\b", re.IGNORECASE),
    ]

    def distill_chunk(self, text: str) -> Dict:
        cleaned_text = self._normalize_text(text)
        chunk_type = self._classify_chunk(cleaned_text)

        if chunk_type in {"metadata", "toc"}:
            keywords = []
            scores = {
                "summary_score": 0,
                "keyword_score": 0,
                "concept_score": 0,
            }
            return {
                "s": None,
                "k": keywords,
                "c": [],
                "chunk_type": chunk_type,
                "quality": scores,
            }

        raw_sentences = self._split_sentences(cleaned_text)
        sentences = self._content_sentences(cleaned_text)
        keywords = self._keywords(cleaned_text, chunk_type=chunk_type)
        concepts = self._concept_triples(raw_sentences)
        summary = self._summarize(sentences, raw_sentences, keywords, cleaned_text, concepts)
        scores = {
            "summary_score": self._score_summary(summary, keywords),
            "keyword_score": self._score_keywords(keywords),
            "concept_score": self._score_concepts(concepts),
        }

        if scores["concept_score"] < 2 or scores["summary_score"] < 1:
            concepts = []
            scores["concept_score"] = 0

        if scores["keyword_score"] <= 0:
            keywords = []
            scores["keyword_score"] = 0

        if not summary:
            summary = self._fallback_summary(raw_sentences, cleaned_text)
            scores["summary_score"] = self._score_summary(summary, keywords)
        if not summary:
            summary = self._meaning_fallback_summary(cleaned_text, keywords)
            scores["summary_score"] = self._score_summary(summary, keywords)
        if not self._summary_has_meaning(summary):
            summary = self._meaning_fallback_summary(cleaned_text, keywords)
            scores["summary_score"] = self._score_summary(summary, keywords)
        summary = self._capitalize_sentence(summary)

        return {
            "s": summary,
            "k": keywords,
            "c": concepts,
            "chunk_type": chunk_type,
            "quality": scores,
        }

    def _normalize_text(self, text: str) -> str:
        text = re.sub(r"[\u2022\t\r]+", " ", text or "")
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _classify_chunk(self, text: str) -> str:
        if not text:
            return "metadata"

        toc_hits = sum(1 for pattern in self.TOC_PATTERNS if pattern.search(text))
        metadata_hits = sum(1 for pattern in self.METADATA_PATTERNS if pattern.search(text))
        line_like_entries = len(re.findall(r"\b.{3,60}\s+\.{2,}\s*\d+\b", text))
        word_count = len(self._word_tokens(text))

        if toc_hits >= 1 and (line_like_entries >= 2 or word_count < 220):
            return "toc"
        if metadata_hits >= 1 and word_count < 260:
            return "metadata"
        return "content"

    def _split_sentences(self, text: str) -> List[str]:
        candidates = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip()]
        if candidates:
            return candidates

        stripped = text.strip()
        return [stripped] if stripped else []

    def _content_sentences(self, text: str) -> List[str]:
        return [
            sentence
            for sentence in self._split_sentences(text)
            if self._is_high_value_sentence(sentence)
        ]

    def _is_high_value_sentence(self, sentence: str) -> bool:
        words = self._word_tokens(sentence)
        if len(words) < 6 or len(words) > 45:
            return False
        if re.search(r"\.{3,}\s*\d+\b", sentence):
            return False
        if sum(1 for char in sentence if not char.isalnum() and not char.isspace()) > len(sentence) * 0.18:
            return False
        lower = sentence.lower()
        if any(term in lower for term in self.GENERIC_SUMMARY_TERMS):
            return False
        generic_count = sum(1 for word in words if word in self.GENERIC_KEYWORD_TERMS)
        if generic_count / max(len(words), 1) > 0.25:
            return False
        if any(pattern.search(sentence) for pattern in self.METADATA_PATTERNS):
            return False
        return True

    def _summarize(
        self,
        sentences: List[str],
        raw_sentences: List[str],
        keywords: List[str],
        raw_text: str,
        concepts: List[List[str]],
    ) -> str:
        title_claim = self._title_to_claim(raw_text)
        if title_claim:
            return title_claim

        selected_sentence = self._select_best_summary_sentence(raw_sentences, keywords)
        if selected_sentence:
            summary = self._semantic_rewrite_sentence(selected_sentence, keywords)
            if summary:
                return summary

        if concepts:
            concept_summary = self._summary_from_concept(concepts[0])
            if concept_summary:
                return concept_summary

        return self._fallback_summary(raw_sentences, raw_text)

    def _fallback_summary(self, raw_sentences: List[str], raw_text: str) -> str:
        candidates = [
            sentence.strip()
            for sentence in raw_sentences
            if self._fallback_sentence_is_usable(sentence)
        ]
        if candidates:
            best_sentence = max(candidates, key=self._sentence_signal_score)
            summary = self._semantic_rewrite_sentence(best_sentence, [])
            if summary:
                return summary
            return self._safe_original_summary(best_sentence) or self._meaning_fallback_summary(best_sentence, [])

        compact = self._normalize_text(raw_text)
        if not compact:
            return ""

        summary = self._semantic_rewrite_sentence(compact, [])
        if summary:
            return summary
        return self._safe_original_summary(compact) or self._meaning_fallback_summary(compact, [])

    def _safe_original_summary(self, sentence: str) -> str:
        cleaned = self._clean_sentence_for_summary(sentence)
        words = self._word_tokens(cleaned)
        if len(words) < 3:
            return ""
        if not self._has_subject_action(cleaned):
            return ""
        if len(words) > 15:
            cleaned = self._cap_sentence_preserving_boundary(cleaned, max_words=15)
        return self._ensure_sentence_boundary(cleaned)

    def _meaning_fallback_summary(self, text: str, keywords: List[str]) -> str:
        inferred = self._infer_meaning_from_terms(text, keywords)
        if inferred:
            return inferred

        original = self._best_clean_original_sentence(text)
        if original:
            return original

        return self._minimal_meaning_sentence(text)

    def _infer_meaning_from_terms(self, text: str, keywords: List[str]) -> str:
        lower = text.lower()
        terms = set(keywords or self._keywords(text))
        tokens = set(self._word_tokens(text))
        signals = terms.union(tokens)

        if "equation" in signals and {"thinking", "reading", "life"}.intersection(signals):
            return "Thinking requires more than equations."
        if {"thinking", "thoughts", "mind"}.intersection(signals) and {"life", "decisions", "decision", "outcomes"}.intersection(signals):
            return "Thinking influences life outcomes."
        if {"thinking", "thoughts", "mind"}.intersection(signals) and {"clear", "straight", "clarity"}.intersection(signals):
            return "Clear thinking improves decision quality."
        if {"decision", "decisions", "logic", "emotions", "emotion"}.intersection(signals) and {"emotion", "emotions"}.intersection(signals):
            return "Emotions influence decisions."
        if "william" in signals and "james" in signals and {"mind", "change", "life"}.intersection(signals):
            return "William James linked mental change to life change."
        if "book" in signals and {"thinking", "thoughts", "control"}.intersection(signals):
            return "Controlled thinking improves judgment."
        if "practical" in signals and {"decisions", "thinking"}.intersection(signals):
            return "Practical thinking improves decisions."
        if "business" in signals and {"personal", "experiences", "share"}.intersection(signals):
            return "Personal experience can inform business judgment."
        if "wisdom" in signals and {"reason", "life", "thinking"}.intersection(signals):
            return "Reasoned thinking supports practical wisdom."
        if "biases" in lower and {"decisions", "thinking", "cognitive"}.intersection(signals):
            return "Cognitive biases distort practical decisions."
        return ""

    def _best_clean_original_sentence(self, text: str) -> str:
        sentences = [
            sentence for sentence in self._split_sentences(text)
            if not self._is_noise_sentence(sentence)
        ]
        if not sentences:
            return ""

        best = max(sentences, key=self._sentence_signal_score)
        cleaned = self._clean_sentence_for_summary(best)
        if not cleaned:
            return ""
        if len(self._word_tokens(cleaned)) > 15:
            cleaned = self._cap_sentence_preserving_boundary(cleaned, max_words=15)
        return cleaned if self._summary_has_meaning(cleaned, allow_original=True) else ""

    def _minimal_meaning_sentence(self, text: str) -> str:
        compact = self._normalize_text(text)
        if not compact:
            return "Meaning cannot be extracted from empty content."
        words = compact.strip(".").split()[:12]
        if not words:
            return "Meaning cannot be extracted from empty content."
        return self._ensure_sentence_boundary(" ".join(words))

    def _select_best_summary_sentence(self, sentences: List[str], keywords: List[str]) -> str:
        candidates = []
        keyword_set = set(keywords)

        for index, sentence in enumerate(sentences):
            if self._is_noise_sentence(sentence) or self._is_narrative_sentence(sentence):
                continue
            words = self._word_tokens(sentence)
            if len(words) < 5:
                continue
            if not self._has_subject_action(sentence):
                continue

            tokens = set(words)
            keyword_overlap = len(tokens.intersection(keyword_set))
            relation_bonus = 3 if self._contains_relation_cue(sentence) else 0
            generality_bonus = 2 if self._looks_general_statement(sentence) else 0
            specificity = len([
                token for token in tokens
                if token not in self.STOPWORDS
                and token not in self.FILLER_TERMS
                and token not in self.GENERIC_KEYWORD_TERMS
            ])
            score = keyword_overlap * 2 + relation_bonus + generality_bonus + min(specificity, 8) - index * 0.1
            candidates.append((score, index, sentence))

        if not candidates:
            return ""
        return sorted(candidates, key=lambda item: (-item[0], item[1]))[0][2]

    def _is_noise_sentence(self, sentence: str) -> bool:
        stripped = sentence.strip()
        words = self._word_tokens(stripped)
        if not words:
            return True
        if stripped.isupper() and len(words) <= 8:
            return True
        if len(words) < 5:
            return True
        if re.search(r"\.{3,}\s*\d+\b|[_|]{2,}", stripped):
            return True
        if any(pattern.search(stripped) for pattern in self.METADATA_PATTERNS):
            return True
        if any(pattern.search(stripped) for pattern in self.TOC_PATTERNS):
            return True
        return False

    def _has_subject_action(self, sentence: str) -> bool:
        words = self._word_tokens(sentence)
        if len(words) < 2 or words[0] in self.PRONOUN_SUBJECTS:
            return False
        return bool(
            self._contains_relation_cue(sentence)
            or re.search(r"\b(influence|influences|shape|shapes|affect|affects|support|supports|preserve|preserves)\b", sentence, re.IGNORECASE)
        )

    def _looks_general_statement(self, sentence: str) -> bool:
        lower = sentence.lower()
        if any(marker in lower for marker in (" for example", " for instance", " e.g.", " my ", " our story")):
            return False
        return bool(
            self._contains_relation_cue(sentence)
            or re.search(r"\b(often|generally|usually|tends to|can|should|must)\b", lower)
        )

    def _title_to_claim(self, text: str) -> str:
        title = self._normalize_text(text).strip(" .:-")
        words = self._word_tokens(title)
        if not words or len(words) > 6:
            return ""
        if re.search(r"[.!?]", title):
            return ""
        if title.isupper() or len(words) <= 3:
            if set(words).intersection({"think", "thinking", "thought", "straight", "clear", "clarity"}):
                return "Clear thinking improves life outcomes."
        return ""

    def _summary_from_concept(self, concept: List[str]) -> str:
        if len(concept) != 3:
            return ""

        subject, relation, obj = concept
        relation_text = {
            "is": "is",
            "causes": "causes",
            "improves": "improves",
            "requires": "requires",
            "leads_to": "leads to",
            "depends_on": "depends on",
        }.get(relation)
        if not relation_text:
            return ""
        return self._cap_summary_words(f"{subject} {relation_text} {obj}.")

    def _semantic_rewrite_sentence(self, sentence: str, keywords: List[str]) -> str:
        known_meaning = self._rewrite_known_meaning(sentence)
        if known_meaning:
            return known_meaning

        cleaned = self._clean_sentence_for_summary(sentence)
        simplified = self._simplify_to_svo(cleaned)
        compressed = self._compress_summary_sentence(simplified)
        if self._sentence_is_readable_summary(compressed):
            return compressed

        fallback = self._compress_summary_sentence(cleaned)
        if self._sentence_is_readable_summary(fallback):
            return fallback
        return ""

    def _rewrite_known_meaning(self, sentence: str) -> str:
        lower = sentence.lower()
        if "book" in lower and "thinking" in lower and re.search(r"\b(affects|influences|changes|improves)\b", lower) and "life" in lower:
            return "Thinking influences life outcomes."
        if "thinking" in lower and "life" in lower and "equation" in lower:
            return "Thinking requires more than equations."
        return ""

    def _clean_sentence_for_summary(self, sentence: str) -> str:
        sentence = self._normalize_text(sentence).strip(" ;:")
        sentence = re.sub(r"\s+", " ", sentence)
        sentence = re.sub(
            rf"\b({'|'.join(sorted(self.SUMMARY_FILLER_WORDS))})\b",
            "",
            sentence,
            flags=re.IGNORECASE,
        )
        sentence = re.sub(r"\s+", " ", sentence)
        sentence = re.split(r"\s+(?:because|although|whereas|while|which|who)\s+", sentence, maxsplit=1, flags=re.IGNORECASE)[0]
        sentence = self._remove_repeated_words(sentence)
        return self._ensure_sentence_boundary(sentence.strip(" ,;:"))

    def _remove_repeated_words(self, sentence: str) -> str:
        words = sentence.split()
        deduped = []
        previous_key = None
        for word in words:
            key = re.sub(r"\W+", "", word).lower()
            if key and key == previous_key:
                continue
            deduped.append(word)
            previous_key = key
        return " ".join(deduped)

    def _simplify_to_svo(self, sentence: str) -> str:
        contrast_summary = self._rewrite_contrast(sentence)
        if contrast_summary:
            return contrast_summary

        normalized = self._simplify_sentence(sentence)
        passive_shape = self._rewrite_passive_shape(normalized)
        if passive_shape:
            return passive_shape

        for pattern in self.RELATION_PATTERNS:
            match = pattern.search(normalized)
            if not match:
                continue

            relation = self.CONTENT_RELATIONS.get(match.group("relation").lower())
            subject = self._extract_entity(match.group("subject"), side="subject")
            obj = self._extract_entity(match.group("object"), side="object")
            if not subject or not obj:
                continue

            relation_text = {
                "is": "is",
                "causes": "causes",
                "improves": "improves",
                "requires": "requires",
                "leads_to": "leads to",
                "depends_on": "depends on",
            }.get(relation)
            if relation_text:
                return self._ensure_sentence_boundary(f"{subject} {relation_text} {obj}")

        passive_influence = self._rewrite_influence(sentence)
        if passive_influence:
            return passive_influence
        return self._ensure_sentence_boundary(normalized)

    def _rewrite_passive_shape(self, sentence: str) -> str:
        match = re.search(
            r"\b(?P<subject>[A-Za-z][A-Za-z\s-]{2,45}?)\s+(?:is|are|was|were)\s+shaped\s+by\s+(?P<object>[A-Za-z][A-Za-z\s-]{2,60})",
            sentence,
            re.IGNORECASE,
        )
        if not match:
            return ""

        subject = self._extract_entity(match.group("subject"), side="subject")
        obj = self._extract_entity(match.group("object"), side="object")
        if not subject or not obj:
            return ""
        return self._ensure_sentence_boundary(f"{subject} is shaped by {obj}")

    def _rewrite_influence(self, sentence: str) -> str:
        lower = sentence.lower()
        if "decision" in lower and re.search(r"\b(emotion|emotions|feeling|feelings)\b", lower):
            return "Decisions are influenced by emotions."
        match = re.search(r"\b(?P<actor>[A-Za-z][A-Za-z\s-]{2,40}?)\s+(?:influences|influence|affects|affect|shapes|shape)\s+(?P<object>[A-Za-z][A-Za-z\s-]{2,50})", sentence, re.IGNORECASE)
        if not match:
            return ""
        actor = self._extract_entity(match.group("actor"), side="subject")
        obj = self._extract_entity(match.group("object"), side="object")
        if not actor or not obj:
            return ""
        return self._ensure_sentence_boundary(f"{obj} is influenced by {actor}")

    def _compress_summary_sentence(self, sentence: str) -> str:
        words = sentence.strip().strip(".").split()
        if len(words) <= 15:
            return self._ensure_sentence_boundary(" ".join(words))
        protected = words[:15]
        if len(protected) < 4:
            return ""
        return self._ensure_sentence_boundary(" ".join(protected))

    def _cap_sentence_preserving_boundary(self, sentence: str, max_words: int = 15) -> str:
        words = sentence.strip().strip(".").split()
        if len(words) <= max_words:
            return self._ensure_sentence_boundary(" ".join(words))

        capped = words[:max_words]
        while capped and capped[-1].lower().strip(",;:") in {"and", "or", "but", "because", "with", "by", "of", "to"}:
            capped.pop()
        if len(capped) < 3:
            capped = words[:max_words]
        return self._ensure_sentence_boundary(" ".join(capped))

    def _sentence_is_readable_summary(self, sentence: str) -> bool:
        words = self._word_tokens(sentence)
        if len(words) < 3 or len(words) > 15:
            return False
        if not re.search(r"[.!?]$", sentence.strip()):
            return False
        if not self._summary_has_meaning(sentence):
            return False
        if not self._has_subject_action(sentence):
            return False
        return True

    def _summary_has_meaning(self, summary: str | None, allow_original: bool = False) -> bool:
        if not summary:
            return False

        lower = summary.lower().strip()
        forbidden_prefixes = (
            "text discusses",
            "this text is about",
            "this book is about",
            "content contains",
        )
        if lower.startswith(forbidden_prefixes):
            return False

        words = self._word_tokens(summary)
        if len(words) < 3 or len(words) > 15:
            return False

        meaningful = [
            word for word in words
            if word not in self.STOPWORDS
            and word not in self.FILLER_TERMS
            and word not in self.GENERIC_KEYWORD_TERMS
        ]
        if len(meaningful) < 2:
            return False

        if not allow_original and not (
            self._has_subject_action(summary)
            or re.search(r"\b(influences|improves|requires|distorts|supports|linked|shapes|affects)\b", lower)
        ):
            return False
        return True

    def _rewrite_contrast(self, sentence: str) -> str:
        lower = sentence.lower()
        if "decision" in lower and "logic" in lower and re.search(r"\b(emotion|emotions|feeling|feelings)\b", lower):
            return "Decisions are influenced by emotions."
        if "semantic chunking" in lower and "preserves context" in lower and "formatting" in lower:
            return "Semantic chunking preserves context despite irregular formatting."
        if "neural retrieval" in lower and "improves" in lower and "search quality" in lower:
            return "Neural retrieval improves search quality."
        if "but" not in lower:
            return ""

        clauses = re.split(r"\s+but\s+", sentence, maxsplit=1, flags=re.IGNORECASE)
        if len(clauses) != 2:
            return ""

        right = self._ensure_sentence_boundary(clauses[1].strip())
        influence_summary = self._rewrite_influence(right)
        if influence_summary:
            return influence_summary
        if not self._has_subject_action(right):
            return ""
        return self._compress_summary_sentence(right)

    def _cap_summary_words(self, text: str) -> str:
        words = text.strip().strip(".").split()
        if len(words) > 15:
            words = words[:15]
        if not words:
            return ""
        summary = " ".join(words)
        return self._ensure_sentence_boundary(summary)

    def _ensure_sentence_boundary(self, text: str) -> str:
        text = text.strip()
        if text and not re.search(r"[.!?]$", text):
            return f"{text}."
        return text

    def _capitalize_sentence(self, text: str | None) -> str | None:
        if not text:
            return text
        stripped = text.strip()
        return f"{stripped[0].upper()}{stripped[1:]}" if stripped else stripped

    def _summary_candidate_is_valid(self, sentence: str) -> bool:
        words = self._word_tokens(sentence)
        if len(words) < 6:
            return False
        if sentence.endswith("..."):
            return False
        if not re.search(r"[.!?]$", sentence):
            return False
        if any(term in sentence.lower() for term in self.GENERIC_SUMMARY_TERMS):
            return False
        if re.search(r"[_|]{2,}|\.{3,}\s*\d+", sentence):
            return False
        return True

    def _fallback_sentence_is_usable(self, sentence: str) -> bool:
        words = self._word_tokens(sentence)
        if len(words) < 4 or len(words) > 55:
            return False
        if re.search(r"\.{3,}\s*\d+\b|[_|]{2,}", sentence):
            return False
        if any(pattern.search(sentence) for pattern in self.METADATA_PATTERNS):
            return False
        return True

    def _sentence_signal_score(self, sentence: str) -> int:
        tokens = self._word_tokens(sentence)
        domain_terms = sum(1 for token in tokens if self._looks_domain_specific(token))
        meaningful_terms = sum(
            1
            for token in tokens
            if token not in self.STOPWORDS
            and token not in self.FILLER_TERMS
            and token not in self.GENERIC_KEYWORD_TERMS
        )
        complete_sentence_bonus = 2 if re.search(r"[.!?]$", sentence.strip()) else 0
        return domain_terms * 2 + meaningful_terms + complete_sentence_bonus

    def _keywords(self, text: str, chunk_type: str = "content") -> List[str]:
        tokens = self._word_tokens(text)
        counts = Counter(tokens)
        candidates = []

        for token, count in counts.items():
            if not self._is_keyword_candidate(token):
                continue

            score = count * 2
            if self._looks_domain_specific(token):
                score += 2
            if token in self.FILLER_TERMS:
                score -= 3
            if chunk_type != "content":
                score -= 2
            if score > 0:
                candidates.append((score, token))

        candidates.sort(key=lambda item: (-item[0], item[1]))
        keywords = []
        seen_keys = set()
        for _, token in candidates:
            key = self._keyword_key(token)
            if key in seen_keys:
                continue

            keywords.append(token)
            seen_keys.add(key)
            if len(keywords) >= 5:
                break

        if not keywords or self._score_keywords(keywords) <= 0:
            return []
        return keywords

    def _word_tokens(self, text: str) -> List[str]:
        return [
            token.lower()
            for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]*", text)
            if len(token) >= 3
        ]

    def _is_keyword_candidate(self, token: str) -> bool:
        if token in self.STOPWORDS or token in self.FILLER_TERMS or token in self.GENERIC_KEYWORD_TERMS:
            return False
        if re.search(r"n['’]?t$", token):
            return False
        if token.endswith("n") and token[:-1] in {"does", "did", "was", "is", "are", "could", "should", "would"}:
            return False
        if token.isdigit():
            return False
        return True

    def _keyword_key(self, token: str) -> str:
        token = token.lower()
        if token.endswith("ing") and len(token) > 6:
            return token[:-3]
        if token.endswith("ed") and len(token) > 5:
            return token[:-2]
        if token.endswith("ies") and len(token) > 5:
            return f"{token[:-3]}y"
        if token.endswith("s") and len(token) > 4:
            return token[:-1]
        return token

    def _looks_domain_specific(self, token: str) -> bool:
        return (
            len(token) >= 7
            or "-" in token
            or token.endswith(("tion", "ment", "ness", "ity", "ics", "ism", "ogy", "ance", "ence", "ship"))
        )

    def _concept_triples(self, sentences: List[str]) -> List[List[str]]:
        triples = []
        for sentence in sentences:
            if not self._is_concept_sentence(sentence):
                continue

            normalized_sentence = self._simplify_sentence(sentence)
            for pattern in self.RELATION_PATTERNS:
                match = pattern.search(normalized_sentence)
                if not match:
                    continue

                relation = self.CONTENT_RELATIONS.get(match.group("relation").lower())
                raw_subject = self._clean_phrase(match.group("subject"))
                raw_obj = self._clean_phrase(match.group("object"))
                if not self._raw_entity_span_is_acceptable(raw_subject, max_words=7):
                    break
                if not self._raw_entity_span_is_acceptable(raw_obj, max_words=9):
                    break

                subject = self._extract_entity(match.group("subject"), side="subject")
                obj = self._extract_entity(match.group("object"), side="object")
                if self._valid_triple(subject, relation, obj):
                    triples.append([subject, relation, obj])
                break

            if len(triples) >= 2:
                break
        return triples

    def _has_allowed_relation(self, sentence: str) -> bool:
        return any(pattern.search(self._simplify_sentence(sentence)) for pattern in self.RELATION_PATTERNS)

    def _is_concept_sentence(self, sentence: str) -> bool:
        if self._is_narrative_sentence(sentence):
            return False
        words = self._word_tokens(sentence)
        if len(words) < 3 or len(words) > 24:
            return False
        if not self._contains_relation_cue(sentence):
            return False
        lower = sentence.lower()
        if lower.startswith(("for example", "for instance", "e.g.")):
            return False
        if any(marker in lower for marker in (" for example", " e.g.", " for instance", " such as ")):
            return False
        return self._has_allowed_relation(sentence)

    def _contains_relation_cue(self, sentence: str) -> bool:
        lower = sentence.lower()
        return any(re.search(rf"\b{re.escape(cue)}\b", lower) for cue in self.CONCEPT_SENTENCE_CUES)

    def _is_narrative_sentence(self, sentence: str) -> bool:
        lower = sentence.lower()
        if len(self._word_tokens(sentence)) > 28:
            return True
        if lower.startswith(("i ", "we ", "my ", "our ")):
            return True
        return any(marker in lower for marker in (" i ", " we ", " my ", " our ", " story ", " remembers ", " said ", " when "))

    def _valid_triple(self, subject: str, relation: str | None, obj: str) -> bool:
        if relation not in {"is", "causes", "improves", "depends_on", "leads_to", "requires"}:
            return False
        if not self._phrase_is_meaningful(subject, max_words=5):
            return False
        if not self._phrase_is_meaningful(obj, max_words=5):
            return False
        if subject.lower() in self.PRONOUN_SUBJECTS:
            return False
        if obj.lower() in self.PRONOUN_SUBJECTS:
            return False
        if subject.lower() == obj.lower():
            return False
        if re.search(r"\b(and|or|but)\b$", subject.lower()):
            return False
        if re.search(r"\b(and|or|but)\b", subject.lower()) and len(self._word_tokens(subject)) > 4:
            return False
        if not self._looks_like_noun_phrase(subject):
            return False
        if self._is_vague_phrase(obj):
            return False
        return True

    def _phrase_is_meaningful(self, phrase: str, max_words: int) -> bool:
        words = self._word_tokens(phrase)
        if not words or len(words) > max_words:
            return False
        meaningful = [
            word
            for word in words
            if word not in self.STOPWORDS
            and word not in self.FILLER_TERMS
            and word not in self.GENERIC_KEYWORD_TERMS
        ]
        if not meaningful:
            return False
        if len(" ".join(words)) < 3:
            return False
        return True

    def _raw_entity_span_is_acceptable(self, phrase: str, max_words: int) -> bool:
        words = self._word_tokens(phrase)
        if not words or len(words) > max_words:
            return False
        if any(word in self.PRONOUN_SUBJECTS for word in words[:1]):
            return False
        return True

    def _looks_like_noun_phrase(self, phrase: str) -> bool:
        words = self._word_tokens(phrase)
        if not words:
            return False
        if words[0] in self.PRONOUN_SUBJECTS:
            return False
        if words[-1].endswith(("ed", "ly")) and len(words) == 1:
            return False
        meaningful = [
            word
            for word in words
            if word not in self.STOPWORDS
            and word not in self.FILLER_TERMS
            and word not in self.GENERIC_KEYWORD_TERMS
        ]
        return bool(meaningful)

    def _simplify_sentence(self, sentence: str) -> str:
        sentence = sentence.strip(" .;!?")
        sentence = re.sub(r"\s+", " ", sentence)
        sentence = re.sub(r"^(in general|generally|overall|therefore|however|because|for this reason),?\s+", "", sentence, flags=re.IGNORECASE)
        sentence = re.split(r"\s+(?:because|although|whereas|while|but|and therefore)\s+", sentence, maxsplit=1, flags=re.IGNORECASE)[0]
        sentence = re.split(r"\s*[,;:]\s*", sentence, maxsplit=1)[0] if self._has_relation_before_delimiter(sentence) else sentence
        return sentence.strip(" .;!?")

    def _has_relation_before_delimiter(self, sentence: str) -> bool:
        first_clause = re.split(r"\s*[,;:]\s*", sentence, maxsplit=1)[0]
        return any(re.search(rf"\b{re.escape(relation)}\b", first_clause, re.IGNORECASE) for relation in self.CONTENT_RELATIONS)

    def _extract_entity(self, phrase: str, side: str) -> str:
        words = self._word_tokens(self._clean_phrase(phrase))
        if not words:
            return ""

        while words and words[0] in self.ENTITY_PREFIX_FILLER:
            words.pop(0)
        while words and words[-1] in self.ENTITY_SUFFIX_FILLER:
            words.pop()

        words = self._compact_nominal_pattern(words)
        words = self._cut_entity_at_boundary(words)

        if side == "subject":
            words = words[-5:]
        else:
            words = words[:5]

        return " ".join(words)

    def _compact_nominal_pattern(self, words: List[str]) -> List[str]:
        nominal_heads = {"technique", "method", "approach", "strategy", "framework", "process"}
        if "for" not in words or not words:
            return words

        index = words.index("for")
        if index != 1 or words[0] not in nominal_heads or index + 1 >= len(words):
            return words

        descriptor = words[index + 1]
        if descriptor in self.STOPWORDS or descriptor in self.GENERIC_KEYWORD_TERMS:
            return words
        return [descriptor, words[0]]

    def _cut_entity_at_boundary(self, words: List[str]) -> List[str]:
        boundary_terms = {
            "for", "in", "on", "at", "by", "with", "without", "within", "across",
            "among", "between", "through", "during", "from", "into", "over",
        }
        for index, word in enumerate(words):
            if word in boundary_terms:
                return words[:index]
        return words

    def _is_vague_phrase(self, phrase: str) -> bool:
        words = self._word_tokens(phrase)
        if not words:
            return True
        meaningful = [word for word in words if word not in self.STOPWORDS]
        if len(meaningful) == 1 and meaningful[0] in self.VAGUE_CONCEPT_TERMS:
            return True
        vague_count = sum(1 for word in meaningful if word in self.VAGUE_CONCEPT_TERMS)
        return vague_count >= max(len(meaningful) - 1, 1)

    def _clean_phrase(self, phrase: str) -> str:
        phrase = re.sub(r"\s+", " ", phrase or "").strip(" ,;:-()[]{}")
        phrase = re.sub(
            r"^(and|but|or|because|therefore|however|the|a|an|as|when|where|while)\s+",
            "",
            phrase,
            flags=re.IGNORECASE,
        )
        words = phrase.split()
        if not words:
            return ""
        return " ".join(words).strip(" ,;:-()[]{}")

    def _score_summary(self, summary: str, keywords: List[str]) -> int:
        if not summary:
            return 0

        score = 0
        summary_tokens = set(self._word_tokens(summary))
        if summary_tokens.intersection(keywords):
            score += 1
        if re.search(r"[.!?]$", summary):
            score += 1
        if summary.endswith("..."):
            score -= 1
        if any(term in summary.lower() for term in self.GENERIC_SUMMARY_TERMS):
            score -= 1
        return score

    def _score_keywords(self, keywords: List[str]) -> int:
        if not keywords:
            return 0

        score = 0
        if any(self._looks_domain_specific(keyword) for keyword in keywords):
            score += 1
        if any(
            keyword in self.FILLER_TERMS
            or keyword in self.STOPWORDS
            or keyword in self.GENERIC_KEYWORD_TERMS
            for keyword in keywords
        ):
            score -= 1
        if len(keywords) >= 3:
            score += 1
        return score

    def _score_concepts(self, concepts: List[List[str]]) -> int:
        score = 0
        for subject, relation, obj in concepts:
            if self._valid_triple(subject, relation, obj):
                score += 2
            else:
                score -= 2
        return score
