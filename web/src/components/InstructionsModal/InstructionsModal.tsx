import Markdown from 'react-markdown';
import { BENCHMARK_DESIGN } from '../../data/benchmarkDesign';
import { InfoModal } from '../InfoModal/InfoModal';

/** Info icon next to the page title; opens a modal with the benchmark's design/methodology
 * writeup. */
export function InstructionsModal() {
  return (
    <InfoModal
      buttonTitle="Benchmark Design"
      ariaLabel="Show benchmark design details"
      heading="Benchmark Design"
      testId="instructions-backdrop"
    >
      <Markdown>{BENCHMARK_DESIGN}</Markdown>
    </InfoModal>
  );
}
