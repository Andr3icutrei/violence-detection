import { Component, OnInit } from '@angular/core';
import { SidebarService } from '../../../../services/sidebar/sidebar.service';
import { DatasetsService } from '../../../../services/datasets/datasets-service';

@Component({
  selector: 'app-inspect-datasets',
  imports: [],
  templateUrl: './inspect-datasets.html',
  styleUrl: './inspect-datasets.css',
})
export class InspectDatasets implements OnInit {
  private readonly pageSize = 10;
  searchTerm: string = '';
  page: number = 1;

  constructor(
    private datasetsService: DatasetsService,
  ) {
  }

  ngOnInit(): void {

  }

  loadDatasets(): void {
    this.datasetsService.getPendingDatasets(this.searchTerm, this.page, this.pageSize).subscribe({

    });
  }
}
