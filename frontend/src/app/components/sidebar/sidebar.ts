import { Component } from '@angular/core';
import { MainLayoutPage } from '../main-layout/main-layout-page.type';
import { NavigationEnd, NavigationExtras, Router } from '@angular/router';
import { filter, Subscription } from 'rxjs';
import { TranslatePipe } from '@ngx-translate/core';
import { SidebarService } from '../../services/sidebar/sidebar.service';

@Component({
  selector: 'app-sidebar',
  imports: [TranslatePipe],
  templateUrl: './sidebar.html',
  styleUrl: './sidebar.css',
})
export class Sidebar {
  selectedSidebarItem: MainLayoutPage | null = 'dashboard';

  constructor(
    private router: Router,
    private sidebarService: SidebarService,
  ) {
    this.sidebarService.refresh$.subscribe((page: MainLayoutPage) => {
      this.syncSelectedSidebarItem(page);
    });
  }

  public goToPage(sidebarItem: MainLayoutPage): void {
    this.selectedSidebarItem = sidebarItem;

    this.router.navigate(['/' + sidebarItem]);
  }

  private syncSelectedSidebarItem(page: MainLayoutPage): void {
    this.selectedSidebarItem = page;
  }
}
