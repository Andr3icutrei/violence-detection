import { Component } from '@angular/core';
import { SidebarService } from '../../../services/sidebar/sidebar.service';

@Component({
  selector: 'app-review-datasets-page',
  imports: [],
  templateUrl: './review-datasets-page.html',
  styleUrl: './review-datasets-page.css',
})
export class ReviewDatasetsPage {
  constructor(private sidebarService: SidebarService) {
    sidebarService.notifyRefresh(null);
  }
}
