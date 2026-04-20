import { Component, OnInit } from '@angular/core';
import { SidebarService } from '../../../services/sidebar/sidebar.service';
import { TranslatePipe } from '@ngx-translate/core';
import { DatasetsService } from '../../../services/datasets/datasets-service';
import { InspectDatasets } from './inspect-datasets/inspect-datasets';

@Component({
  selector: 'app-review-datasets-page',
  imports: [InspectDatasets],
  templateUrl: './review-datasets-page.html',
  styleUrl: './review-datasets-page.css',
})
export class ReviewDatasetsPage {
  constructor(
    private sidebarService: SidebarService,
  ) {
    sidebarService.notifyRefresh('review-datasets');
  }
}
