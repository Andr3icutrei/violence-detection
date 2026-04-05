import { Component } from '@angular/core';
import { MainLayoutPage } from '../main-layout/main-layout-page.type';
import { NavigationEnd, NavigationExtras, Router } from '@angular/router';
import { filter } from 'rxjs';

@Component({
  selector: 'app-sidebar',
  imports: [],
  templateUrl: './sidebar.html',
  styleUrl: './sidebar.css',
})
export class Sidebar {
  selectedSidebarItem: MainLayoutPage | null = 'dashboard';

  constructor(private router: Router) {
    this.syncSelectedSidebarItem();

    this.router.events
      .pipe(filter((event): event is NavigationEnd => event instanceof NavigationEnd))
      .subscribe(() => this.syncSelectedSidebarItem());
  }

  public goToPage(sidebarItem: MainLayoutPage): void {
    this.selectedSidebarItem = sidebarItem;

    const extras: NavigationExtras | undefined =
      sidebarItem === 'inference' ? { state: { fromVideoCard: false } } : undefined;

    this.router.navigate([sidebarItem], extras);
  }

  private syncSelectedSidebarItem(): void {
    const navigationState =
      (this.router.getCurrentNavigation()?.extras.state ?? history.state) as
        | { fromVideoCard?: boolean }
        | undefined;

    if (this.router.url.includes('/inference')) {
      this.selectedSidebarItem = navigationState?.fromVideoCard ? null : 'inference';
      return;
    }

    if (this.router.url.includes('/dashboard')) {
      this.selectedSidebarItem = 'dashboard';
      return;
    }

    this.selectedSidebarItem = null;
  }
}
