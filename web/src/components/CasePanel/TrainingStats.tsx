import Markdown from 'react-markdown';
import { caseTrainingStats } from '../../data/caseText';
import type { CaseMeta, ScoreRow } from '../../types/data';

type Props = {
  scores: ScoreRow[];
  caseMeta: CaseMeta;
  caseId: string;
  methods: string[];
  extended: boolean;
};

export function TrainingStats({ scores, caseMeta, caseId, methods, extended }: Props) {
  return <Markdown>{caseTrainingStats(scores, caseMeta, caseId, methods, extended)}</Markdown>;
}
