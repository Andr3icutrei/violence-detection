import { Component } from '@angular/core';
import { SidebarService } from '../../../services/sidebar/sidebar.service';
import { InspectDatasets } from './inspect-datasets/inspect-datasets';

@Component({
  selector: 'app-datasets-page',
  imports: [InspectDatasets],
  templateUrl: './datasets-page.html',
  styleUrl: './datasets-page.css',
})
export class DatasetsPage {
  constructor(private readonly sidebarService: SidebarService) {
    this.sidebarService.notifyRefresh('datasets');
  }
}
