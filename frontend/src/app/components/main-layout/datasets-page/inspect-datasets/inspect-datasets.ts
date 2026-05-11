import { Component } from '@angular/core';
import { DatasetsPanel } from './datasets-panel/datasets-panel';
import { DatasetsService } from '../../../../services/datasets/datasets-service';

@Component({
  selector: 'app-inspect-datasets',
  imports: [DatasetsPanel],
  templateUrl: './inspect-datasets.html',
  styleUrl: './inspect-datasets.css',
})
export class InspectDatasets {
  constructor(private datasetsService: DatasetsService) {}
}
