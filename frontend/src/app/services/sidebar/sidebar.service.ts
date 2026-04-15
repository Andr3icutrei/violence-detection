import { Injectable } from '@angular/core';
import { Observable, Subject } from 'rxjs';
import { MainLayoutPage } from '../../components/main-layout/main-layout-page.type';

@Injectable({
  providedIn: 'root',
})
export class SidebarService {
  private readonly subject: Subject<MainLayoutPage> = new Subject();

  public get refresh$(): Observable<MainLayoutPage> {
    return this.subject.asObservable();
  }

  public notifyRefresh(page: MainLayoutPage): void {
    this.subject.next(page);
  }
}
