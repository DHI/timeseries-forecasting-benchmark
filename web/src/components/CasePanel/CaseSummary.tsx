import Markdown from 'react-markdown';
import { caseSummary } from '../../data/caseText';
import type { CaseMeta } from '../../types/data';

export function CaseSummary({ caseMeta }: { caseMeta: CaseMeta }) {
  return <Markdown>{caseSummary(caseMeta)}</Markdown>;
}
